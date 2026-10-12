# ForgeCert: certified integer-only compilation

**Author: Vaishnavi**

ForgeCert is ModelForge's main research contribution. It compiles a trained
float neural network to **integer-only C99** for microcontrollers without a
floating-point unit, and ships a **certificate** that a small, independent
checker re-validates from the emitted C file itself. The two earlier
contributions, Guardian-APC+ (per-pass validation of graph rewrites) and
shrink-and-check (falsification of compressed models), are now components of
this pipeline.

```
 .mforge / ONNX ─► front end ─► IR ─► graph passes, each validated by Guardian-APC+
                                        │
                                        ▼
                       ┌──── precision tuner (CEGAR) ◄─────────────┐
                       │   choose bits per weight / activation     │ blame (error provenance)
                       ▼                                           │
            lowering + range inference ─► integer program ─► prover (differential affine AI
            (scales, multipliers/shifts,                        + input splitting)
             accumulator widths)                                   │ fails
                                                                   ▼
                                                           falsifier (real counterexample?)
                       │ all regions proved
                       ▼
       model_q.c / model_q.h (integer-only C99)  +  certificate.json
                       │
                       ▼
       modelforge_certcheck  (independent, shares no code with the compiler)
```

## What is guaranteed

For the input domain `D` (a box, by default the training data's range
widened by 10%) and every certification region `R` (an L∞ ball of radius
`eps` around a real sample, inside `D`):

1. **No overflow.** No accumulator partial sum leaves its C type, and the
   64-bit requantisation product cannot overflow, for any input in `D`.
2. **No saturation.** The input quantiser and every requantised activation
   stay inside their integer range for any input in `D`, so the clamps in the
   code are dead on `D`.
3. **Error bound.** `|scale · q_c(x) − f_c(x)|` is below the certified bound
   for every logit `c` and every `x` in `D`.
4. **Same decision.** For every `x` in `R`, the C function `mf_infer_q`
   returns `argmax f(x)`, where `f` is the float model evaluated in exact real
   arithmetic.

Claims 1 to 4 are about the bit-exact semantics of the emitted C: float32
input scaling, round-half-away-from-zero, packed 4-bit weights, integer
accumulation, ReLU on the accumulator, `round(acc · m / 2^s)` requantisation
and first-index tie-breaking in the argmax. They are not about an idealised
"fake-quantised" float model.

## How it works

### Integer target and range inference (`src/qnet.*`, `src/diffai.cpp`)

Every dense layer becomes

```c
acc = bias[o] + Σ_i W[o][i] · x[i]           // int16 / int32 / int64, inferred
if (relu && acc < 0) acc = 0;
y[o] = clamp(round_shift(acc · M[o], S[o]))  // fixed-point requantisation
```

Weights are quantised per output channel after folding the input channel
scales in (`W[o,i] · inScale[i]`), so per-channel activation scales cost
nothing at run time. Lowering is interleaved with the analysis below: each
layer's output scale is chosen from the *proved* range of its values over
`D`, and its accumulator width from the proved range of its inputs. If a
multiplier rounding could push a value past its range, the scale is grown
and the layer is re-checked.

### Differential affine abstract interpretation (`src/diffai.*`)

Every neuron carries two affine forms over shared noise symbols:
`R` (the float model, DeepZ ReLU relaxation) and `E` (error of the integer
program: `scale · q − R`). The affine step is exact for both:

```
R' = W R + b
E' = (Ŵ − W) R + Ŵ E + (b̂ − b)      Ŵ, b̂: effective real weights of the emitted integers
```

For ReLU, the error `ReLU(R+E) − ReLU(R)` is bounded by the tighter of
(a) the difference of the two DeepZ relaxations, which keeps correlations,
and (b) a ReluDiff-style interval `conv{0, E}` intersected with what each
stable case implies. Both stable cases are handled exactly. Rounding adds a
fresh symbol of radius `scale/2`.

`E` is further split by **provenance**, with one component per error source:
input quantisation, each layer's weights (incl. bias and multiplier), and
each layer's output rounding. The split is exact through affine steps and
proportional through relaxations. The proofs use the sum. The parts tell the
tuner which layer to blame.

A region is proved when `lb(R_c − R_j) > 0` (the float decision is constant)
and `lb((R_c+E_c) − (R_j+E_j)) > 0` for all rivals `j` (with half an LSB more
for lower-index rivals, because of the argmax tie rule). Otherwise the region
is bisected along the input dimension with the largest coefficient in the
failing margin, up to a depth and analysis budget.

### Counterexample-guided precision tuning (`src/certify.*`)

```
config ← cheapest rung everywhere (4-bit weights, 8-bit activations)
loop
    lower config (range inference)
    for every region: falsify cheaply; if no counterexample, prove (with splitting)
    if all proved: break
    blame_s ← Σ over failing regions of source s's share of the failing margin
    raise the source with the largest blame_s / Δbytes_s by one rung
minimise: lower any width whose removal keeps every region proved
```

The falsifier starts from the vertex that the abstract margin points to
(the sign pattern of its input coefficients), then runs an adaptive local
search on the integer program. A counterexample it reports is a real input,
in the region and replayable, on which the emitted C disagrees with the float
model.

### Independent checker (`tools/certcheck.cpp`)

`modelforge_certcheck <dir> <model.mforge>` includes no ModelForge header. It

* parses the source model and the data arrays of `model_q.c`, and checks
  every value against its declared C type and bit width;
* regenerates the code template from the metadata and checks that the C
  file contains exactly that code;
* re-derives claims 1 to 3 over `D` with its own implementation of the
  analysis;
* replays each region's split tree (from the certificate) and proves every
  leaf;
* optionally (`--expect raw.csv inputs.csv`) compares its own integer
  semantics with the output of the binary that a real C compiler produced.

The compiler is large and does search. The checker is small and does none:
it only replays the leaves it is given. This is the proof-carrying-code split
between finding a proof and checking it.

## Results

Results are in `ModelForge/research/results/certbench.json`; summarise them with
`python ModelForge/research/summarize_certbench.py ModelForge/research/results/certbench.json`.


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

What did **not** pan out, reported in the paper too:

* On the coarse byte lattice (4/8/16-bit), provenance-guided refinement ties
  with "raise every layer". Provenance pays off on the fine bit-level lattice:
  10% smaller programs than raising everything, and 13–23% smaller than random
  choice.
* Range inference proves overflow freedom, but it never gave a narrower
  accumulator than a range-agnostic compiler on these models.

## Commands

```bash
# certify and emit integer-only C (exit code 4 if some region stays unproved)
modelforge model.mforge --certify --data test.csv --domain train.csv --eps 0.1 --out build/cert
# check a fixed precision instead of tuning
modelforge model.mforge --certify --data test.csv --precision int8 --out build/cert8
# independent re-check, including the compiled binary
cmake -S build/cert -B build/cert_build && cmake --build build/cert_build
build/cert_build/generated_model_q --raw test.csv > build/cert/raw.csv
modelforge_certcheck build/cert model.mforge --expect build/cert/raw.csv test.csv
# full evaluation (paper tables)
modelforge_certbench --out build/certbench
```

## Limits

* Dense ReLU networks (Gemm/MatMul+Add, ReLU, output Softmax/Sigmoid).
  Convolutions are future work: they lower to the same integer dot product,
  but the analysis cost grows with the feature-map size.
* The reference is the float model in exact real arithmetic. Float32
  rounding inside the original framework is not modelled.
* The analysis and the checker use double precision with outward slack, not
  interval arithmetic with directed rounding.
* The domain-wide error bound (claim 3) is loose. The region claims (4) are
  what matter for deployment.
* A region on which the float model itself is not provably constant is
  excluded and reported, not certified.
* The checker re-implements the analysis independently, but was written by
  the same author as the compiler.

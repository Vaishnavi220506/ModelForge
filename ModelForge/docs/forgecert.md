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

For the input domain `D` (a box: the training data's range widened by 10%,
or `--domain-range 0,1` for images) and every certification region `R` (an
L∞ ball of radius `eps` around a real sample, inside `D`):

1. **No overflow.** No product leaves its multiply type, no accumulator
   leaves its C type, and the 64-bit requantisation product cannot overflow,
   for any input in `D`. A classical accumulator needs every partial sum to
   fit. A *modular* accumulator wraps around in unsigned arithmetic, so only
   the final sum must fit, and the abstract interpretation bounds that sum.
2. **Saturation.** With `--scales proved` (the default) no clamp can trigger
   on `D`. With `--scales calibrated` (scales from data, as TFLite-style flows
   choose them) clamps can trigger, and every proof models them exactly.
3. **Error bound.** `|scale · q_c(x) − f_c(x)|` is below the certified bound
   for every logit `c` and every `x` in `D`.
4. **Same decision.** For every `x` in `R`, the C function `mf_infer_q`
   returns `argmax f(x)`, where `f` is the float model evaluated in exact real
   arithmetic.

These claims are about the bit-exact semantics of the emitted C:

* float32 input scaling, with round-half-away-from-zero;
* packed 4-bit weights;
* integer and modular accumulation;
* ReLU on the accumulator, then `round(acc · m / 2^s)` and the clamp;
* first-index tie-breaking in the argmax.

They are not about an idealised "fake-quantised" float model.

## How it works

### Integer target and proof-guided lowering (`src/qnet.*`, `src/diffai.cpp`)

Every linear layer, dense or 2-D convolution, is a list of *taps*
(input, weight) per output neuron. It becomes

```c
acc = bias[c] + Σ (mul_t)W[w] · (mul_t)x[i]  // int16/int32/int64 or modular uint16/uint32
if (relu && acc < 0) acc = 0;
y[o] = clamp(round_shift(acc · M[c], S[c]))   // fixed-point requantisation per channel
```

with convolutions emitted as sliding-window loops over weight tables in flash
(`MF_ROM`, i.e. `__flash` on AVR). Lowering is interleaved with the analysis
over the whole domain. Every width below is a *proof obligation discharged at
compile time* and re-checked by the checker:

| Decision | From |
|---|---|
| activation scales (per channel) | proved range (`proved`) or data (`calibrated`, saturation modelled) |
| accumulator width, classical | bound on every partial sum |
| accumulator width, **modular** | bound on the *final* sum (tighter); partial sums may wrap |
| **16-bit products** | max \|weight\| · max \|activation\| ≤ 32767 |
| **dead-input elimination** | input proved 0 on `D` in both float and integer program → its weights are emitted as 0 and do not distort the scale |
| unsigned input codes | non-negative domain (doubles input resolution) |

### Differential affine abstract interpretation (`src/diffai.*`)

Every neuron carries two affine forms over shared noise symbols:
`R` (the float model, DeepZ ReLU relaxation) and `E` (the error of the integer
program, `scale · q − R`). The affine step is exact for both:

```
R' = W R + b
E' = (Ŵ − W) R + Ŵ E + (b̂ − b)      Ŵ, b̂: effective real weights of the emitted integers
```

**ReLU.** The error `ReLU(R+E) − ReLU(R)` is exact in both stable cases.
Otherwise it is bounded by the tighter of two forms: (a) the difference of the
two DeepZ relaxations, which keeps correlations, and (b) a ReluDiff-style
interval `conv{0, E}` intersected with what each half-stable case implies.

**Clamp.** Saturation `clamp(Y) − Y = −ReLU(Y − U)` (or `ReLU(−U − Y)`) is
relaxed like a ReLU, so it keeps correlations as well.

**Rounding** adds a fresh symbol of radius `scale/2`, except for values that
are provably 0 or provably at the clamp: those round exactly.

`E` can be split by **provenance**, with one component per error source:
input quantisation, each layer's weights, and each layer's rounding and
saturation. The tuner uses this split to decide which layer to make more
precise. It is computed only at a leaf where a proof fails, so proofs run with
a single error form. Regions are proved in parallel.

### Counterexample-guided precision tuning (`src/certify.*`)

```
config ← cheapest rung everywhere
loop
    lower config (proof-guided lowering)
    for every region: analyse; if not proved, falsify (real counterexample?), else split and prove
    if all proved: break
    blame_s ← Σ over failing regions of source s's share of the failing margin
    raise the source with the largest blame_s / Δcost_s by one rung
local search: any one-rung move (down, up, or to the next storage class)
              that keeps every region proved and lowers the cost
```

There are three objectives:

* `--cost bytes`: flash + RAM, with 4/8/16-bit weights and 8/16-bit activations.
* `--cost bits`: ideal packed size on the fine lattice.
* `--cost latency`: estimated cycles on an 8-bit AVR, on the fine lattice. The
  cost per multiply-accumulate is measured with simavr for every combination
  of weight storage, activation width, multiply width and accumulator width.

The latency objective is where the proved widths pay off. For example,
6-bit weights × 10-bit activations still need only a 16-bit multiply, which
on AVR costs half as much as a 32-bit one.

### Independent checker (`tools/certcheck.cpp`)

`modelforge_certcheck <dir> <model.mforge>` includes no ModelForge header. It

* parses the source model, dense or convolutional, and the data arrays of
  `model_q.c`, and checks every value against its declared C type and bit
  width;
* regenerates the code template (dense and convolution loops, classical or
  modular accumulation, multiply type) from the metadata and checks that the
  C file contains exactly that code;
* re-derives the domain-wide claims with its own implementation of the
  analysis, including the clamp model: products, partial or final sums, and
  saturation if claimed;
* replays each region's split tree from the certificate and proves every leaf;
* optionally (`--expect raw.csv inputs.csv`) compares its own integer
  semantics with the output of the binary a real C compiler produced.

### Running on a microcontroller (`research/mcu_bench.py`)

The emitted C99 compiles unchanged with `avr-gcc` and `arm-none-eabi-gcc`.
`mcu_bench.py` builds every model for an ATmega2560 (Arduino Mega: 8-bit,
16 MHz, 8 KB RAM, no FPU) and runs it in the cycle-accurate simulator
simavr. It checks that every prediction equals the host build of the same C
file, and records cycles, flash and RAM next to the float32 C baseline and
the Cortex-M0 code size.

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
# an image CNN, calibrated scales, tuned for latency on an 8-bit MCU
modelforge mnist_cnn.mforge --certify --data mnist_test.csv --domain mnist_calib.csv \
    --domain-range 0,1 --eps 0.01 --scales calibrated --cost latency --out build/cert_cnn
# check a fixed precision instead of tuning
modelforge model.mforge --certify --data test.csv --precision int8 --out build/cert8
# independent re-check, including the compiled binary
cmake -S build/cert -B build/cert_build && cmake --build build/cert_build
build/cert_build/generated_model_q --raw test.csv > build/cert/raw.csv
modelforge_certcheck build/cert model.mforge --expect build/cert/raw.csv test.csv
# full evaluation (paper tables) and the microcontroller study
modelforge_certbench --suite all --out build/certbench
python ModelForge/research/mcu_bench.py build --samples 10 --out mcu_results.json
```

## Limits

* ReLU networks built from dense layers and 2-D convolutions (no pooling,
  batch norm must be folded into the weights, no residual connections).
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

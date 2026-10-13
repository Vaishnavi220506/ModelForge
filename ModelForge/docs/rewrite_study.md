# Validating compiler rewrites on real models (`modelforge_rewritebench`)

Author: Vaishnavi

How quickly does a validation method notice that a graph rewrite broke a model?
This study runs **21 trained models** (tabular MLPs, MNIST/Fashion MLPs and CNNs,
a CIFAR-10 CNN, LeNet-5, MobileNet-style and ResNet-style CNNs) through
**22 correct rewrites** (negative controls) and **179 faulty rewrites** that
reproduce documented compiler bugs (`src/rewrites.cpp`):

| Rewrite | Faulty variant | Documented source |
| --- | --- | --- |
| Conv + BatchNorm folding | wrong argument used as epsilon | PyTorch #102224 |
| Conv + BatchNorm folding | scale applied on the wrong weight axis | PyTorch #70022 |
| Conv + BatchNorm folding / constant folding | constants rounded to half precision | fp16 conversion |
| Activation lowering | ReLU lowered as ReLU6 | ONNX Runtime #9753, #10936 |
| Activation lowering | quantised-ReLU threshold off by 0.05 | quantisation |
| Gemm + ReLU fusion / Add + ReLU fusion | activation dropped / applied to one operand | TFLite #57818 |
| Dead-code elimination | live residual branch removed | ONNX Runtime #13425 |
| Conv padding | padded cells not zero | ONNX Runtime #11828, PyTorch #86556 |
| NHWC layout before Flatten | dense weights not permuted | TVM #10593, #10156 |
| ReLU / pooling commute | moved past average pooling (control: max pooling) | synthetic |

**Equal budgets.** Every method gets the same number of *model evaluations*
(checking a probe = 2, a gradient = 2, a forward pass = 1). Baselines: random,
validation data, data + noise, TensorFuzz-style coverage-guided fuzzing,
DeepXplore-style neuron ascent, DiffChaser-style genetic differential search.
Guardian-APC+ = smoke probes, boundary-value probes of ReLU units along exact
gradients, rewrite-aware unit selection, near-miss search.

**Independent oracle.** The original model runs in an independent float64
implementation; every witness is re-checked with ONNX Runtime
(`research/rewrite_oracle.py`, its own ONNX converter).

**Results** (`research/results/rewrites/summary.txt`, 5 seeds):

* 0 false positives on the 22 correct rewrites (1,100 runs); ONNX Runtime
  confirms all 158 witnesses.
* 69 of the 179 faulty rewrites are *loud*: the first random and the first data
  input both find them. Validation strategy does not matter for these.
* On the 89 *subtle* faults, at 512 evaluations: Guardian-APC+ 85.2%,
  DiffChaser-style 79.1%, DeepXplore-style 75.3%, validation data 73.5%,
  TensorFuzz-style 66.5%, random 62.9%. Model-level Wilcoxon vs the genetic
  search p = 0.033 (10 models better, 5 worse); vs data p = 0.0012. At 64
  evaluations it ties with data and the genetic search.
* Ablation: boundary-value probing exposes activation-lowering faults, near-miss
  search exposes numeric faults; **rewrite-aware targeting gave no measurable
  gain** on these models.

```text
modelforge_rewritebench --budget 512 --seeds 5 --out build/rw
python ModelForge/research/rewrite_oracle.py build/rw
python ModelForge/research/analyze_rewrites.py build/rw
```

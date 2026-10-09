# ModelForge demo samples

Author: **Vaishnavi**

Everything here is ready to show: one command builds the compiler, compiles a
sample neural network to C++ twice (without and with optimisation), and shows
what changed and that the answers stay the same.

## Files

| File | What it is |
|---|---|
| `three_layer_demo.mforge` | A three-layer classifier (6 inputs, 3 classes) written so that all three optimisations apply: constant folding, layer fusion and dead-code removal. |
| `inputs.csv` | 8 test inputs (6 numbers per row) for the generated programs. |
| `run_demo.ps1` | The full demo for Windows PowerShell. |
| `run_demo.sh` | The same demo for Linux and macOS. |

## Run it

Windows (VS Code terminal, from the repository folder):

```powershell
.\samples\run_demo.ps1
```

Linux or macOS:

```bash
./samples/run_demo.sh
```

## What the demo shows

1. **Compile without optimisation**: `modelforge ... --no-opt` writes the
   plain C++ to `build/demo_plain/`.
2. **Compile with Guardian-checked optimisation**: the compiler folds the
   constant bias addition, fuses each Gemm with its ReLU, and removes the unused
   `debug_probe` node. Guardian tests every one of those rewrites before it is
   kept. Instructions go from **10 to 6**.
3. **The generated `infer()` side by side**. Plain version:

   ```cpp
   Tensor v_b1 = add(c_b1_base, c_b1_offset);
   Tensor v_z1 = gemm(v_x, c_w1, &v_b1, 1, 6, 8, false, false);
   Tensor v_h1 = relu(v_z1);
   Tensor v_z2 = gemm(v_h1, c_w2, &c_b2, 1, 8, 8, false, false);
   Tensor v_h2 = relu(v_z2);
   Tensor v_logits = gemm(v_h2, c_w3, &c_b3, 1, 8, 3, false, false);
   Tensor v_y = softmax(v_logits);
   Tensor v_debug_value = sigmoid(v_h1);
   ```

   Optimised version:

   ```cpp
   Tensor v_h1 = fused_gemm_relu(v_x, c_w1, &c_b1, 1, 6, 8, false, false);
   Tensor v_h2 = fused_gemm_relu(v_h1, c_w2, &c_b2, 1, 8, 8, false, false);
   Tensor v_logits = gemm(v_h2, c_w3, &c_b3, 1, 8, 3, false, false);
   Tensor v_y = softmax(v_logits);
   ```

   Fewer steps and fewer temporary tensors; the folded bias is baked in as a
   constant and the dead `sigmoid` is gone.
4. **Both programs are built and run on `inputs.csv`**: the predictions and
   scores are identical.
5. **Guardian catches a broken rewrite**: a deliberately wrong optimisation
   (ReLU turned into Sigmoid) is rejected at the first targeted input, and the
   report names the layer where the outputs first differ (`activation_1`).
6. **Shrink and check**: a real Wine classifier compressed to float16, int8,
   int4 and pruned. int4 keeps 98.1% accuracy with zero test-set changes, but
   Guardian finds a realistic input where the models disagree, and recommends
   int8.

## Try it by hand

```powershell
.\build\Release\modelforge.exe samples\three_layer_demo.mforge --out build\mine --dump-cpp
cmake -S build\mine -B build\mine\build
cmake --build build\mine\build --config Release
.\build\mine\build\Release\generated_model.exe 1 -2 0.5 3 0 -1
.\build\mine\build\Release\generated_model.exe --csv samples\inputs.csv
```

The interactive website (`website/index.html`, section **Compiler**) does the
same in the browser: pick this sample, press **Compile**, and look at the
generated C++, the before/after comparison, the inspector and the run panel.

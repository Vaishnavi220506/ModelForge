#!/usr/bin/env bash
# ModelForge terminal demo (by Vaishnavi).
#
#   ./ModelForge/demo/demo.sh            full demo (~40 s)
#   ./ModelForge/demo/demo.sh --fast     skip the CNN and the microcontroller (~10 s)
#   ./ModelForge/demo/demo.sh --pause    wait for Enter between steps (for presenting)
#
# Steps
#   1  the problem: plain int4/int8 code changes the model's decisions
#   2  ForgeCert: counterexample-guided precision tuning with proofs
#   3  independent checker + bit-exact run of the generated C
#   4  a CNN (LeNet-5 with max-pooling) certified, against uniform int8
#   5  the certified code on a simulated 8-bit microcontroller (simavr)
#   6  Guardian-APC+: catching wrong compiler rewrites
#   7  rewrite study: Guardian-APC+ against fuzzing baselines at equal budget
#   8  headline results of the full evaluation
set -uo pipefail  # no -e: baseline runs exit non-zero by design (unproved regions)

FAST=0
PAUSE=0
for a in "$@"; do
    case "$a" in
        --fast) FAST=1 ;;
        --pause) PAUSE=1 ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown option $a"; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CORE="$ROOT/ModelForge/review_2_core_implementation"
RESEARCH="$ROOT/ModelForge/research"
BIN="$ROOT/build/Release"
OUT="$ROOT/build/demo"
REAL="$CORE/models/real"
IMG="$CORE/models/image"

if [ -t 1 ]; then
    B=$'\e[1m'; D=$'\e[2m'; R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; C=$'\e[36m'; N=$'\e[0m'
else
    B=""; D=""; R=""; G=""; Y=""; C=""; N=""
fi

step() {
    echo
    echo "${B}${C}== $1 ${N}"
    [ -n "${2:-}" ] && echo "${D}$2${N}"
    return 0
}
run() {
    echo "${Y}\$ ${*//$ROOT\//}${N}" >&2
    "$@"
}
# Same, with the tool's stderr (e.g. the --trace stream) merged into stdout.
run_all() {
    echo "${Y}\$ ${*//$ROOT\//}${N}" >&2
    "$@" 2>&1
}
pause() {
    if [ "$PAUSE" = 1 ]; then read -r -p "${D}[Enter]${N}" _; fi
    return 0
}
ok() { echo "${G}${B}$*${N}"; }
bad() { echo "${R}${B}$*${N}"; }
# Strip the absolute output path from compiler logs.
tidy() { sed -e "s#$ROOT/##g"; }

cat <<EOF
${B}
  __  __           _      _ _____
 |  \/  | ___   __| | ___| |  ___|__  _ __ __ _  ___
 | |\/| |/ _ \ / _\` |/ _ \ |  |_ / _ \| '__/ _\` |/ _ \\
 | |  | | (_) | (_| |  __/ |  _| (_) | | | (_| |  __/
 |_|  |_|\___/ \__,_|\___|_|_|  \___/|_|  \__, |\___|
                                          |___/${N}
  ForgeCert: proof-carrying integer code for neural networks on microcontrollers
  Guardian-APC+: validation of ML compiler rewrites          ${D}by Vaishnavi${N}
EOF

# ---------------------------------------------------------------- build
if [ ! -x "$BIN/modelforge" ] || [ ! -x "$BIN/modelforge_certcheck" ] || [ ! -x "$BIN/modelforge_rewritebench" ]; then
    step "Building ModelForge (Release)"
    run cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
    run cmake --build "$ROOT/build" -j"$(nproc)" >/dev/null
fi
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$CORE"

# ---------------------------------------------------------------- 1
step "1. The problem: quantised code can change the model's decision" \
     "Digits classifier (64 inputs, 10 classes). Plain int4 code, the usual way, then a search for inputs
within L-inf 0.1 of each test image on which the integer code decides differently from the float model."
run "$BIN/modelforge" "$REAL/digits_mlp16.mforge" --certify --precision int4 \
    --data "$REAL/digits_test.csv" --domain "$REAL/digits_train.csv" --eps 0.1 --max-regions 60 \
    --no-ir --out "$OUT/int4" | grep -E 'Integer program|flash|regions:' | tidy
CEX="$OUT/int4/counterexamples.csv"
if [ -f "$CEX" ]; then
    LINE="$(sed -n 2p "$CEX")"
    REGION="$(cut -d, -f1 <<<"$LINE")"
    INPUT="$(cut -d, -f4- <<<"$LINE")"
    echo
    echo "Replaying counterexample ${B}$REGION${N} in the generated C, compiled with cc:"
    cc -O2 -std=c99 -o "$OUT/int4/model_int" "$OUT/int4/main_q.c" "$OUT/int4/model_q.c"
    cat >"$OUT/int4/main_f.c" <<'C'
#include <stdio.h>
#include <stdlib.h>
#include "model_f.h"
int main(int argc, char **argv) {
    float x[64], y[16];
    for (int i = 1; i < argc && i <= 64; ++i) x[i - 1] = strtof(argv[i], NULL);
    printf("Prediction: %d\n", mf_predict_f(x, y));
    return 0;
}
C
    cc -O2 -std=c99 -o "$OUT/int4/model_float" "$OUT/int4/main_f.c" "$OUT/int4/model_f.c"
    IFS=, read -r -a X <<<"$INPUT"
    printf "  float32 model : %s\n" "$("$OUT/int4/model_float" "${X[@]}" | grep Prediction)"
    printf "  int4 code     : %s\n" "$("$OUT/int4/model_int" "${X[@]}" | grep Prediction)"
    bad "  -> same input, different decision. Test accuracy does not reveal this."
fi
pause

# ---------------------------------------------------------------- 2
step "2. ForgeCert: certified integer compilation" \
     "Counterexample-guided tuning: prove every region with differential abstract interpretation; where a proof
fails, blame the layer that causes it and raise only that layer's precision."
run_all "$BIN/modelforge" "$REAL/digits_mlp16.mforge" --certify --trace \
    --data "$REAL/digits_test.csv" --domain "$REAL/digits_train.csv" --eps 0.1 --max-regions 60 \
    --no-ir --out "$OUT/cert" | sed -n '/\[3\/6\]/,$p' | grep -v 'Re-check' | tidy
ok "  -> every robust region proved, in less flash than float32"
pause

# ---------------------------------------------------------------- 3
step "3. Trust nothing: independent checker and bit-exact execution" \
     "A separate program (no code shared with the compiler) re-derives every proof from the emitted C file,
then compares its integer semantics with the compiled binary on all test rows."
cc -O2 -std=c99 -o "$OUT/cert/model_q" "$OUT/cert/main_q.c" "$OUT/cert/model_q.c"
"$OUT/cert/model_q" --raw "$REAL/digits_test.csv" >"$OUT/cert/raw.txt"
echo "${Y}\$ cc main_q.c model_q.c && ./model_q --raw digits_test.csv > raw.txt${N}"
echo "  $(wc -l <"$OUT/cert/raw.txt") predictions with integer logits, e.g. $(head -1 "$OUT/cert/raw.txt")"
run "$BIN/modelforge_certcheck" "$OUT/cert" "$REAL/digits_mlp16.mforge" \
    --expect "$OUT/cert/raw.txt" "$REAL/digits_test.csv" | tidy
pause

# ---------------------------------------------------------------- 4
if [ "$FAST" = 0 ]; then
    step "4. Beyond MLPs: LeNet-5 (conv, max-pool) on MNIST, radius one grey level" \
         "Uniform int8 first, then ForgeCert with calibrated scales (saturation is modelled in the proofs)."
    echo "${B}uniform int8:${N}"
    run "$BIN/modelforge" "$IMG/mnist_lenet.mforge" --certify --precision int8 \
        --data "$IMG/mnist_test.csv" --domain "$IMG/mnist_calib.csv" --domain-range 0,1 \
        --eps 0.00392157 --max-regions 10 --no-ir --out "$OUT/lenet_int8" | grep -E 'flash|regions:' | tidy
    echo "${B}ForgeCert:${N}"
    run_all "$BIN/modelforge" "$IMG/mnist_lenet.mforge" --certify --scales calibrated --trace \
        --data "$IMG/mnist_test.csv" --domain "$IMG/mnist_calib.csv" --domain-range 0,1 \
        --eps 0.00392157 --max-regions 10 --no-ir --out "$OUT/lenet" \
        | grep -E 'tuner|improve|Integer program|flash|regions:' | tidy
    run "$BIN/modelforge_certcheck" "$OUT/lenet" "$IMG/mnist_lenet.mforge" | tail -1
    pause
fi

# ---------------------------------------------------------------- 5
if [ "$FAST" = 0 ]; then
    step "5. On the device: 8-bit AVR, no FPU (cycle-accurate simavr)"
    if command -v avr-gcc >/dev/null && command -v simavr >/dev/null; then
        run python3 "$RESEARCH/mcu_demo.py" "$OUT/cert" "$REAL/digits_test.csv" --samples 5
    else
        echo "  skipped: needs avr-gcc and simavr (apt install gcc-avr avr-libc simavr)"
    fi
    pause
fi

# ---------------------------------------------------------------- 6
step "6. Guardian-APC+: is an optimised/converted model still the same model?" \
     "A hand-optimised Iris model with a tiny bug in one weight (error 0.001), then a compiler pass with a wrong
ReLU rewrite. Guardian searches activation boundaries for an input that tells them apart."
run "$BIN/modelforge" models/iris_demo.mforge --compare models/iris_handopt_bug.mforge --no-ir --out "$OUT/compare" \
    | grep -E 'differ|RESULT|input:|reference:|candidate:|First difference' | tidy
echo
run "$BIN/modelforge" models/iris_demo.mforge --guardian-demo-bug --no-ir --out "$OUT/guardian" \
    | grep -E 'REJECTED|localised|Guardian (constant|dense|dead)|Translation' | tidy
pause

# ---------------------------------------------------------------- 7
step "7. Rewrite study: catching documented compiler bugs" \
     "Real rewrites (Conv+BN folding, GEMM+ReLU fusion, padding, layout, fp16 constants...) with bugs reproduced
from established ML compilers, plus correct versions as controls. Every method gets the same budget of model
evaluations; a detection counts only if a float64 oracle confirms it. Live run on one model (5 seeds x 512):"
run "$BIN/modelforge_rewritebench" --only digits_mlp32x16 --out "$OUT/rewrites" | grep -v '^done' | tidy
echo
echo "${B}Full study (21 models, 179 faulty + 22 correct rewrites, 5 seeds), from research/results/rewrites:${N}"
SUMMARY="$RESEARCH/results/rewrites/summary.txt"
sed -n '1,3p' "$SUMMARY"
echo
sed -n '5,6p' "$SUMMARY"
grep -E '^(random|data|tensorfuzz|deepxplore|diffchaser|guardian_apc_plus) +0\.' "$SUMMARY"
echo
grep -E '^  (random|data|tensorfuzz|deepxplore|diffchaser)@512' "$SUMMARY" \
    | sed -E 's/ours-only.*\| models/models/; s/^  /  Guardian-APC+ vs /'
pause

# ---------------------------------------------------------------- 8
step "8. Headline results (full evaluation, see README and paper/)"
cat <<EOF
  ForgeCert   tabular  ${B}1440/1441${N} regions certified; exhaustive optimum on 7/8 models (gap 0.3%)
              image    ${B}671/676${N} certified
              CNN zoo  uniform int8 proves 21/345 (147 real counterexamples); ForgeCert ${B}343/345${N}
              ${B}4,458,800${N} audit executions inside proved regions: ${G}${B}0 violations${N}
              independent checker: ${B}41/41${N} certificates valid
  MCU         every program that fits is bit-exact on the AVR; float32 is 2.2-2.3x slower;
              certified ResNet runs on an ATmega1284P (16/18 proved, int8 proves 0/18)
  Guardian    0 false positives in 1,100 control runs; ONNX Runtime confirms 158/158 witnesses;
              subtle bugs at 512 evaluations: ${B}85.2%${N} vs DiffChaser 79.1%, DeepXplore 75.3%,
              data 73.5%, TensorFuzz 66.5%, random 62.9%
EOF
echo
ok "Demo finished. Outputs in ${OUT#$ROOT/}/"

#!/usr/bin/env bash
# ModelForge faculty demo (author: Vaishnavi).
# Compiles the same model without and with Guardian-checked optimisation,
# shows the difference in the generated C++, proves both programs give the
# same answers, and shows Guardian rejecting a broken rewrite.
set -e
cd "$(dirname "$0")/.."
step() { printf '\n\033[1;34m== %s\033[0m\n' "$1"; }

step "Build ModelForge"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build --config Release > /dev/null
MF=build/Release/modelforge

step "1. Compile WITHOUT optimisation"
$MF samples/three_layer_demo.mforge --out build/demo_plain --no-opt --no-ir | grep -E "Loaded|Optimization|Generated"

step "2. Compile WITH Guardian-checked optimisation"
$MF samples/three_layer_demo.mforge --out build/demo_opt --no-ir | grep -E "Loaded|Instructions|Removed|Rewrite|Guardian |PASS|Generated"

step "3. The generated infer() function: left = plain, right = optimised"
infer() { sed -n '/^std::vector<float> infer/,/^}/p' "$1"; }
diff -y -W 200 <(infer build/demo_plain/model.cpp) <(infer build/demo_opt/model.cpp) || true

step "4. Build both generated programs"
for v in plain opt; do
  cmake -S build/demo_$v -B build/demo_$v/build -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build build/demo_$v/build --config Release > /dev/null
done
exe() { if [ -x "build/demo_$1/build/generated_model" ]; then echo "build/demo_$1/build/generated_model"; else echo "build/demo_$1/build/Release/generated_model"; fi; }

step "5. Same answers? Both programs on samples/inputs.csv"
"$(exe plain)" --csv samples/inputs.csv > build/demo_plain/predictions.csv
"$(exe opt)" --csv samples/inputs.csv > build/demo_opt/predictions.csv
cat build/demo_opt/predictions.csv
if cmp -s build/demo_plain/predictions.csv build/demo_opt/predictions.csv; then
  echo "IDENTICAL predictions and scores from both programs."
else
  echo "Outputs differ:"; diff build/demo_plain/predictions.csv build/demo_opt/predictions.csv || true
fi

step "6. Guardian catches a broken rewrite (ReLU wrongly turned into Sigmoid)"
$MF ModelForge/review_2_core_implementation/models/iris_demo.mforge --out build/demo_bug --guardian-demo-bug --no-ir | grep -E "Guardian demo|localised"
cat build/demo_bug/counterexample.json

step "7. Shrink and check a real trained model"
$MF ModelForge/review_2_core_implementation/models/real/wine_mlp16.mforge --shrink all \
    --data ModelForge/review_2_core_implementation/models/real/wine_test.csv --out build/demo_shrink | sed -n '5,20p'

# ModelForge faculty demo for Windows PowerShell (author: Vaishnavi).
# Run from the repository folder:  .\samples\run_demo.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }

Step "Build ModelForge"
cmake -S . -B build | Out-Null
cmake --build build --config Release | Out-Null
$MF = ".\build\Release\modelforge.exe"

Step "1. Compile WITHOUT optimisation"
& $MF samples\three_layer_demo.mforge --out build\demo_plain --no-opt --no-ir | Select-String "Loaded|Optimization|Generated"

Step "2. Compile WITH Guardian-checked optimisation"
& $MF samples\three_layer_demo.mforge --out build\demo_opt --no-ir | Select-String "Loaded|Instructions|Removed|Rewrite|Guardian |PASS|Generated"

Step "3. The generated infer() function"
foreach ($v in "plain", "opt") {
  Write-Host "--- $v ---" -ForegroundColor Yellow
  $lines = Get-Content "build\demo_$v\model.cpp"
  $start = ($lines | Select-String "^std::vector<float> infer").LineNumber - 1
  $lines[$start..($lines.Length - 1)] | ForEach-Object { $_; if ($_ -eq "}") { break } }
}

Step "4. Build both generated programs"
foreach ($v in "plain", "opt") {
  cmake -S "build\demo_$v" -B "build\demo_$v\build" | Out-Null
  cmake --build "build\demo_$v\build" --config Release | Out-Null
}

Step "5. Same answers? Both programs on samples\inputs.csv"
& "build\demo_plain\build\Release\generated_model.exe" --csv samples\inputs.csv | Set-Content build\demo_plain\predictions.csv
& "build\demo_opt\build\Release\generated_model.exe" --csv samples\inputs.csv | Set-Content build\demo_opt\predictions.csv
Get-Content build\demo_opt\predictions.csv
$diff = Compare-Object (Get-Content build\demo_plain\predictions.csv) (Get-Content build\demo_opt\predictions.csv)
if ($diff) { Write-Host "Outputs differ:" -ForegroundColor Red; $diff } else { Write-Host "IDENTICAL predictions and scores from both programs." -ForegroundColor Green }

Step "6. Guardian catches a broken rewrite (ReLU wrongly turned into Sigmoid)"
& $MF ModelForge\review_2_core_implementation\models\iris_demo.mforge --out build\demo_bug --guardian-demo-bug --no-ir | Select-String "Guardian demo|localised"
Get-Content build\demo_bug\counterexample.json

Step "7. Shrink and check a real trained model"
& $MF ModelForge\review_2_core_implementation\models\real\wine_mlp16.mforge --shrink all --data ModelForge\review_2_core_implementation\models\real\wine_test.csv --out build\demo_shrink

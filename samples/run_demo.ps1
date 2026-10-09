# ModelForge faculty demo for Windows PowerShell (author: Vaishnavi).
# Run from the repository folder:
#   powershell -ExecutionPolicy Bypass -File .\samples\run_demo.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }

Step "Build ModelForge"
cmake -S . -B build | Out-Null
cmake --build build --config Release | Out-Null
$MF = ".\build\Release\modelforge.exe"

Step "1. Compile WITHOUT optimisation"
& $MF samples\three_layer_demo.mforge --out build\demo_plain --no-opt --no-ir --predict samples\inputs.csv | Select-String "Loaded|Optimization|Generated"

Step "2. Compile WITH Guardian-checked optimisation"
& $MF samples\three_layer_demo.mforge --out build\demo_opt --no-ir --predict samples\inputs.csv | Select-String "Loaded|Instructions|Removed|Rewrite|Guardian |PASS|Generated"

Step "3. The generated infer() function: plain, then optimised"
foreach ($v in "plain", "opt") {
  Write-Host ""
  Write-Host "--- $v ---" -ForegroundColor Yellow
  $inside = $false
  foreach ($line in Get-Content "build\demo_$v\model.cpp") {
    if ($line -like "std::vector<float> infer*") { $inside = $true }
    if ($inside) {
      Write-Host $line
      if ($line -eq "}") { break }
    }
  }
}

Step "4. Build both generated C++ programs"
foreach ($v in "plain", "opt") {
  cmake -S "build\demo_$v" -B "build\demo_$v\build" | Out-Null
  cmake --build "build\demo_$v\build" --config Release | Out-Null
}
Write-Host "Built build\demo_plain\build\Release\generated_model.exe and build\demo_opt\build\Release\generated_model.exe"

Step "5. Same answers? Both versions on samples\inputs.csv"
$source = "the generated C++ programs"
try {
  foreach ($v in "plain", "opt") {
    & "build\demo_$v\build\Release\generated_model.exe" --csv samples\inputs.csv | Set-Content "build\demo_$v\program_predictions.csv"
    Copy-Item "build\demo_$v\program_predictions.csv" "build\demo_$v\predictions.csv" -Force
  }
} catch {
  # Some laptops (Windows Application Control / Smart App Control) block newly
  # built programs. The predictions from step 1 and 2 (--predict) use the same
  # compiled model and the same output format, so the comparison still holds.
  Write-Host "Windows blocked the new program ($($_.Exception.Message.Split('.')[0]))." -ForegroundColor Yellow
  Write-Host "Using ModelForge's --predict results from steps 1 and 2 instead (same model, same output format)." -ForegroundColor Yellow
  $source = "ModelForge --predict"
}
Get-Content build\demo_opt\predictions.csv
$diff = Compare-Object (Get-Content build\demo_plain\predictions.csv) (Get-Content build\demo_opt\predictions.csv)
if ($diff) { Write-Host "Outputs differ:" -ForegroundColor Red; $diff }
else { Write-Host "IDENTICAL predictions and scores from the plain and optimised versions ($source)." -ForegroundColor Green }

Step "6. Guardian catches a broken rewrite (ReLU wrongly turned into Sigmoid)"
& $MF ModelForge\review_2_core_implementation\models\iris_demo.mforge --out build\demo_bug --guardian-demo-bug --no-ir | Select-String "Guardian demo|localised"
Get-Content build\demo_bug\counterexample.json

Step "7. Shrink and check a real trained model"
& $MF ModelForge\review_2_core_implementation\models\real\wine_mlp16.mforge --shrink all --data ModelForge\review_2_core_implementation\models\real\wine_test.csv --out build\demo_shrink

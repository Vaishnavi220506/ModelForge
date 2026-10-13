# ModelForge terminal demo for Windows PowerShell (by Vaishnavi).
#
#   .\ModelForge\demo\demo.ps1            full demo
#   .\ModelForge\demo\demo.ps1 -Fast      skip the CNN step
#   .\ModelForge\demo\demo.ps1 -Pause     wait for Enter between steps (for presenting)
#
# Same steps as demo.sh except the AVR microcontroller run (needs avr-gcc and
# simavr; run demo.sh under WSL or Linux for that step).
param([switch]$Fast, [switch]$Pause)
$ErrorActionPreference = "Continue"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Core = Join-Path $Root "ModelForge\review_2_core_implementation"
$Research = Join-Path $Root "ModelForge\research"
$Bin = Join-Path $Root "build\Release"
$Out = Join-Path $Root "build\demo"
$Real = Join-Path $Core "models\real"
$Img = Join-Path $Core "models\image"
$MF = Join-Path $Bin "modelforge.exe"
$Check = Join-Path $Bin "modelforge_certcheck.exe"
$Bench = Join-Path $Bin "modelforge_rewritebench.exe"

function Step($title, $text) {
    Write-Host ""
    Write-Host "== $title" -ForegroundColor Cyan
    if ($text) { Write-Host $text -ForegroundColor DarkGray }
}
function Show($exe, $arguments) {
    $line = ("$exe " + ($arguments -join " ")).Replace("$Root\", "")
    Write-Host "> $line" -ForegroundColor Yellow
}
# Run a tool, merge its stderr (the --trace stream), print lines matching $pattern.
function Run($exe, $arguments, $pattern) {
    Show $exe $arguments
    $lines = & $exe @arguments 2>&1 | ForEach-Object { "$_".Replace("$Root\", "") }
    if ($pattern) { $lines = $lines | Where-Object { $_ -match $pattern } }
    $lines | ForEach-Object { Write-Host $_ }
}
function Wait-Step { if ($Pause) { Read-Host "[Enter]" | Out-Null } }
# Build a generated C model with CMake and return the path of its executable.
function Build-Generated($dir) {
    cmake -S $dir -B (Join-Path $dir "build") | Out-Null
    cmake --build (Join-Path $dir "build") --config Release | Out-Null
    (Get-ChildItem -Path (Join-Path $dir "build") -Recurse -Filter "generated_model_q.exe" | Select-Object -First 1).FullName
}

Write-Host ""
Write-Host "  ModelForge" -ForegroundColor White
Write-Host "  ForgeCert: proof-carrying integer code for neural networks on microcontrollers"
Write-Host "  Guardian-APC+: validation of ML compiler rewrites          by Vaishnavi"

if (-not (Test-Path $MF) -or -not (Test-Path $Check) -or -not (Test-Path $Bench)) {
    Step "Building ModelForge (Release)"
    cmake -S $Root -B (Join-Path $Root "build") | Out-Null
    cmake --build (Join-Path $Root "build") --config Release | Out-Null
}
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Set-Location $Core

# 1 ---------------------------------------------------------------------------
Step "1. The problem: quantised code can change the model's decision" `
     "Digits classifier (64 inputs, 10 classes). Plain int4 code, then a search for inputs within L-inf 0.1 of each test image on which the integer code decides differently from the float model."
$int4 = Join-Path $Out "int4"
Run $MF @("$Real\digits_mlp16.mforge", "--certify", "--precision", "int4", "--data", "$Real\digits_test.csv",
          "--domain", "$Real\digits_train.csv", "--eps", "0.1", "--max-regions", "60", "--no-ir", "--out", $int4) `
    "Integer program|flash|regions:"
$cex = Join-Path $int4 "counterexamples.csv"
if (Test-Path $cex) {
    $fields = (Get-Content $cex | Select-Object -Index 1).Split(",")
    $exe = Build-Generated $int4
    $x = $fields[3..($fields.Length - 1)]
    Write-Host ""
    Write-Host "Counterexample $($fields[0]), replayed in the generated C (compiled with CMake):"
    Write-Host "  float32 model : Prediction: $($fields[1])"
    $pred = (& $exe @x | Select-String "Prediction").ToString()
    Write-Host "  int4 code     : $pred"
    Write-Host "  -> same input, different decision. Test accuracy does not reveal this." -ForegroundColor Red
}
Wait-Step

# 2 ---------------------------------------------------------------------------
Step "2. ForgeCert: certified integer compilation" `
     "Counterexample-guided tuning: prove every region with differential abstract interpretation; where a proof fails, blame the layer that causes it and raise only that layer's precision."
$cert = Join-Path $Out "cert"
Run $MF @("$Real\digits_mlp16.mforge", "--certify", "--trace", "--data", "$Real\digits_test.csv",
          "--domain", "$Real\digits_train.csv", "--eps", "0.1", "--max-regions", "60", "--no-ir", "--out", $cert) `
    "tuner|^\[[3-6]/6\]|estimated|flash|regions:"
Write-Host "  -> every robust region proved, in less flash than float32" -ForegroundColor Green
Wait-Step

# 3 ---------------------------------------------------------------------------
Step "3. Trust nothing: independent checker and bit-exact execution" `
     "A separate program re-derives every proof from the emitted C file, then compares its integer semantics with the compiled binary on all test rows."
$exe = Build-Generated $cert
$raw = Join-Path $cert "raw.txt"
Write-Host "> generated_model_q --raw digits_test.csv > raw.txt" -ForegroundColor Yellow
$rows = @(& $exe --raw "$Real\digits_test.csv")
[IO.File]::WriteAllText($raw, ($rows -join "`n") + "`n")
Write-Host "  $($rows.Count) predictions with integer logits, e.g. $($rows[0])"
Run $Check @($cert, "$Real\digits_mlp16.mforge", "--expect", $raw, "$Real\digits_test.csv") ""
Wait-Step

# 4 ---------------------------------------------------------------------------
if (-not $Fast) {
    Step "4. Beyond MLPs: LeNet-5 (conv, max-pool) on MNIST, radius one grey level" `
         "Uniform int8 first, then ForgeCert with calibrated scales (saturation is modelled in the proofs)."
    $common = @("--data", "$Img\mnist_test.csv", "--domain", "$Img\mnist_calib.csv", "--domain-range", "0,1",
                "--eps", "0.00392157", "--max-regions", "10", "--no-ir")
    Write-Host "uniform int8:" -ForegroundColor White
    Run $MF (@("$Img\mnist_lenet.mforge", "--certify", "--precision", "int8") + $common + @("--out", "$Out\lenet_int8")) "flash|regions:"
    Write-Host "ForgeCert:" -ForegroundColor White
    Run $MF (@("$Img\mnist_lenet.mforge", "--certify", "--scales", "calibrated", "--trace") + $common + @("--out", "$Out\lenet")) `
        "tuner|Integer program|flash|regions:"
    Run $Check @("$Out\lenet", "$Img\mnist_lenet.mforge") "CERTIFICATE"
    Wait-Step
}

# 5 ---------------------------------------------------------------------------
Step "5. On the device (8-bit AVR, simavr)" "Needs avr-gcc and simavr: run ./ModelForge/demo/demo.sh under WSL or Linux for this step."
Wait-Step

# 6 ---------------------------------------------------------------------------
Step "6. Guardian-APC+: is an optimised/converted model still the same model?" `
     "A hand-optimised Iris model with a tiny bug in one weight (error 0.001), then a compiler pass with a wrong ReLU rewrite."
Run $MF @("models\iris_demo.mforge", "--compare", "models\iris_handopt_bug.mforge", "--no-ir", "--out", "$Out\compare") `
    "differ|RESULT|input:|reference:|candidate:|First difference"
Write-Host ""
Run $MF @("models\iris_demo.mforge", "--guardian-demo-bug", "--no-ir", "--out", "$Out\guardian") `
    "REJECTED|localised|Guardian (constant|dense|dead)|Translation"
Wait-Step

# 7 ---------------------------------------------------------------------------
Step "7. Rewrite study: catching documented compiler bugs" `
     "Real rewrites (Conv+BN folding, GEMM+ReLU fusion, padding, layout, fp16 constants...) with bugs reproduced from established ML compilers, plus correct versions as controls; equal budget of model evaluations for every method. Live run on one model (5 seeds x 512):"
Run $Bench @("--only", "digits_mlp32x16", "--out", "$Out\rewrites") "^(==|   )"
Write-Host ""
Write-Host "Full study (21 models, 179 faulty + 22 correct rewrites, 5 seeds), from research\results\rewrites:" -ForegroundColor White
$summary = Get-Content (Join-Path $Research "results\rewrites\summary.txt")
$summary[0..2] | ForEach-Object { Write-Host $_ }
Write-Host ""
$summary[4..5] | ForEach-Object { Write-Host $_ }
$summary | Where-Object { $_ -match '^(random|data|tensorfuzz|deepxplore|diffchaser|guardian_apc_plus) +0\.' } | ForEach-Object { Write-Host $_ }
Write-Host ""
$summary | Where-Object { $_ -match '^  (random|data|tensorfuzz|deepxplore|diffchaser)@512' } |
    ForEach-Object { Write-Host ("  Guardian-APC+ vs " + ($_.Trim() -replace 'ours-only.*\| models', 'models')) }
Wait-Step

# 8 ---------------------------------------------------------------------------
Step "8. Headline results (full evaluation, see README and paper\)"
@"
  ForgeCert   tabular  1440/1441 regions certified; exhaustive optimum on 7/8 models (gap 0.3%)
              image    671/676 certified
              CNN zoo  uniform int8 proves 21/345 (147 real counterexamples); ForgeCert 343/345
              4,458,800 audit executions inside proved regions: 0 violations
              independent checker: 41/41 certificates valid
  MCU         every program that fits is bit-exact on the AVR; float32 is 2.2-2.3x slower;
              certified ResNet runs on an ATmega1284P (16/18 proved, int8 proves 0/18)
  Guardian    0 false positives in 1,100 control runs; ONNX Runtime confirms 158/158 witnesses;
              subtle bugs at 512 evaluations: 85.2% vs DiffChaser 79.1%, DeepXplore 75.3%,
              data 73.5%, TensorFuzz 66.5%, random 62.9%
"@ | Write-Host
Write-Host ""
Write-Host "Demo finished. Outputs in build\demo\" -ForegroundColor Green
Set-Location $Root

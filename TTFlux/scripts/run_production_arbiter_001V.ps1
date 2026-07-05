param(
    [switch]$SkipBatch,
    [string]$RunDir = ".\runs\batch_001E",
    [string]$ConfigDir = ".\configs\batch_001E"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== TTFlux production runner 001V ==="
Write-Host "[001V] RunDir    : $RunDir"
Write-Host "[001V] ConfigDir : $ConfigDir"
Write-Host "[001V] SkipBatch : $SkipBatch"

$runnerArgs = @(
    ".\scripts\run_batch_with_arbiter_001T2.py",
    "--run-dir", $RunDir,
    "--config-dir", $ConfigDir
)

if ($SkipBatch) {
    $runnerArgs = @(
        ".\scripts\run_batch_with_arbiter_001T2.py",
        "--skip-batch",
        "--run-dir", $RunDir,
        "--config-dir", $ConfigDir
    )
}

Write-Host ""
Write-Host "[001V] Step 1/2 : build clean features via 001T2"
python @runnerArgs

$microCsv = Join-Path $RunDir "micro_blob_features_001T2.csv"
$outDir = Join-Path $RunDir "arbiter_001V"

if (!(Test-Path $microCsv)) {
    throw "[001V] Missing micro features: $microCsv"
}

Write-Host ""
Write-Host "[001V] Step 2/2 : apply production arbiter 001U"
python .\tools\apply_production_arbiter_001U.py `
    --features $microCsv `
    --out-dir $outDir

$html = Join-Path $outDir "arbiter_001U.html"
$json = Join-Path $outDir "arbiter_001U_summary.json"

Write-Host ""
Write-Host "[001V] DONE"
Write-Host "[001V] final HTML : $html"
Write-Host "[001V] final JSON : $json"

if (Test-Path $html) {
    ii $html
}

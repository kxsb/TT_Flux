param(
    [string]$RunDir = ".\runs\batch_001E",
    [string]$ArbiterDir = ".\runs\batch_001E\arbiter_001V",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== TTFlux review queue 001W ==="
Write-Host "[001W] RunDir     : $RunDir"
Write-Host "[001W] ArbiterDir : $ArbiterDir"

$reviewCsv = Join-Path $ArbiterDir "arbiter_001U_review.csv"
$mediaDir = Join-Path $RunDir "review_media_001W"
$outRawHtml = Join-Path $RunDir "video_review_001W_raw.html"
$outFrHtml = Join-Path $RunDir "video_review_001W_fr.html"

if (!(Test-Path $reviewCsv)) {
    throw "[001W] Missing review CSV: $reviewCsv"
}

$rows = Import-Csv $reviewCsv
Write-Host "[001W] review rows:" $rows.Count

if ($rows.Count -eq 0) {
    throw "[001W] No review rows to display."
}

$forceArgs = @()
if ($Force) {
    $forceArgs = @("--force")
}

Write-Host ""
Write-Host "[001W] Step 1/2 : build/transcode video review"
python .\tools\build_video_review_001H2.py `
    --run-dir $RunDir `
    --manifest $reviewCsv `
    --out-html $outRawHtml `
    --media-dir $mediaDir `
    @forceArgs

Write-Host ""
Write-Host "[001W] Step 2/2 : polish French review UI"
python .\tools\polish_video_review_001H3_fr.py `
    --run-dir $RunDir `
    --manifest $reviewCsv `
    --media-dir $mediaDir `
    --out-html $outFrHtml

Write-Host ""
Write-Host "[001W] DONE"
Write-Host "[001W] review CSV : $reviewCsv"
Write-Host "[001W] raw HTML   : $outRawHtml"
Write-Host "[001W] final HTML : $outFrHtml"

if (Test-Path $outFrHtml) {
    ii $outFrHtml
}

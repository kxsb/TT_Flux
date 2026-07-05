$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = (Get-Location).Path
$TtnetRoot = Join-Path $Root "data_external\code\ttnet_pytorch"
$RawOpenTT = Join-Path $Root "data_external\raw\openttgames"
$ExtractedOpenTT = Join-Path $Root "data_external\extracted\openttgames"
$Dataset = Join-Path $TtnetRoot "dataset"

$TrainNames = @("game_1","game_2","game_3","game_4","game_5")
$TestNames  = @("test_1","test_2","test_3","test_4","test_5","test_6","test_7")

function Ensure-Dir($p) {
    New-Item -ItemType Directory -Force -Path $p | Out-Null
}

function Copy-IfExists($src, $dst) {
    if (Test-Path $src) {
        Ensure-Dir (Split-Path -Parent $dst)
        Copy-Item -LiteralPath $src -Destination $dst -Force
        Write-Host "OK copy $src -> $dst"
        return $true
    }
    return $false
}

function Find-AnnotationDir($name) {
    $candidates = @(
        (Join-Path $ExtractedOpenTT $name),
        (Join-Path $ExtractedOpenTT "$name\$name"),
        (Join-Path $RawOpenTT $name)
    )

    foreach ($c in $candidates) {
        if ((Test-Path $c) -and ((Test-Path (Join-Path $c "ball_markup.json")) -or (Test-Path (Join-Path $c "events_markup.json")) -or (Test-Path (Join-Path $c "segmentation_masks")))) {
            return $c
        }
    }

    $hit = Get-ChildItem -Path $ExtractedOpenTT -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq $name -and (
                (Test-Path (Join-Path $_.FullName "ball_markup.json")) -or
                (Test-Path (Join-Path $_.FullName "events_markup.json")) -or
                (Test-Path (Join-Path $_.FullName "segmentation_masks"))
            )
        } |
        Select-Object -First 1

    if ($hit) { return $hit.FullName }
    return $null
}

function Sync-One($name, $split) {
    $annDst = Join-Path $Dataset "$split\annotations\$name"
    $vidDst = Join-Path $Dataset "$split\videos\$name.mp4"
    $imgDst = Join-Path $Dataset "$split\images\$name"

    Ensure-Dir $annDst
    Ensure-Dir (Split-Path -Parent $vidDst)
    Ensure-Dir $imgDst

    $annSrc = Find-AnnotationDir $name
    if ($annSrc) {
        Copy-Item -LiteralPath (Join-Path $annSrc "*") -Destination $annDst -Recurse -Force
        Write-Host "OK annotations $name"
    } else {
        Write-Warning "Annotations introuvables pour $name"
    }

    $vidSrc = Join-Path $RawOpenTT "$name.mp4"
    if (!(Copy-IfExists $vidSrc $vidDst)) {
        Write-Warning "Vidéo absente pour $name : $vidSrc"
    }
}

Ensure-Dir $Dataset

foreach ($name in $TrainNames) {
    Sync-One $name "training"
}

foreach ($name in $TestNames) {
    Sync-One $name "test"
}

Write-Host ""
Write-Host "=== Résumé TTNet dataset ==="
Get-ChildItem -Path $Dataset -Recurse -File |
    Select-Object FullName, Length |
    Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $Dataset "_ttnet_dataset_inventory.csv")

Write-Host "Inventory : data_external\code\ttnet_pytorch\dataset\_ttnet_dataset_inventory.csv"

Write-Host ""
Write-Host "Vidéos training :"
Get-ChildItem "$Dataset\training\videos" -Filter *.mp4 -ErrorAction SilentlyContinue | Select-Object Name,Length

Write-Host ""
Write-Host "Vidéos test :"
Get-ChildItem "$Dataset\test\videos" -Filter *.mp4 -ErrorAction SilentlyContinue | Select-Object Name,Length

Write-Host ""
Write-Host "Annotations training :"
Get-ChildItem "$Dataset\training\annotations" -Directory -ErrorAction SilentlyContinue | Select-Object Name

Write-Host ""
Write-Host "Annotations test :"
Get-ChildItem "$Dataset\test\annotations" -Directory -ErrorAction SilentlyContinue | Select-Object Name

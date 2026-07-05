$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = (Get-Location).Path
$Base = Join-Path $Root "data_external"
$RawOpenTT = Join-Path $Base "raw\openttgames"
$ExtractedOpenTT = Join-Path $Base "extracted\openttgames"
$TtnetRoot = Join-Path $Base "code\ttnet_pytorch"
$Dataset = Join-Path $TtnetRoot "dataset"
$Log = Join-Path $Base "_openttgames_ttnet_sync_006C.log"

$TrainNames = @("game_1","game_2","game_3","game_4","game_5")
$TestNames  = @("test_1","test_2","test_3","test_4","test_5","test_6","test_7")
$AllNames   = $TrainNames + $TestNames

$OpenTTBase = "https://lab.osai.ai/datasets/openttgames/data"

function Log($m) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m"
    $line | Tee-Object -FilePath $Log -Append
}

function Ensure-Dir($p) {
    New-Item -ItemType Directory -Force -Path $p | Out-Null
}

function Has-Cmd($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Download-File($url, $out) {
    Ensure-Dir (Split-Path -Parent $out)

    if ((Test-Path $out) -and ((Get-Item $out).Length -gt 1000)) {
        Log "SKIP exists $out"
        return
    }

    Log "DOWNLOAD $url -> $out"

    if (Has-Cmd "curl.exe") {
        & curl.exe -L --fail --retry 5 --retry-delay 2 -o "$out" "$url"
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed: $url"
        }
    } else {
        Invoke-WebRequest -Uri $url -OutFile $out
    }
}

function Expand-Zip($zip, $dest) {
    Ensure-Dir $dest
    Log "EXTRACT $zip -> $dest"

    try {
        Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
    } catch {
        if (Has-Cmd "7z") {
            & 7z x "$zip" "-o$dest" -y
            if ($LASTEXITCODE -ne 0) {
                throw "7z failed: $zip"
            }
        } else {
            throw "Extraction failed and 7z missing: $zip"
        }
    }
}

function Find-ExistingFile($name, $ext) {
    $hit = Get-ChildItem -Path $Base -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq "$name$ext" } |
        Select-Object -First 1

    if ($hit) { return $hit.FullName }
    return $null
}

function Find-AnnotationDir($name) {
    $hit = Get-ChildItem -Path $Base -Recurse -Directory -ErrorAction SilentlyContinue |
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

Ensure-Dir $RawOpenTT
Ensure-Dir $ExtractedOpenTT
Ensure-Dir $Dataset

if (Test-Path $Log) { Remove-Item $Log -Force }

Log "START 006C"

foreach ($name in $AllNames) {
    $zipPath = Join-Path $RawOpenTT "$name.zip"
    $mp4Path = Join-Path $RawOpenTT "$name.mp4"

    if (!(Test-Path $zipPath)) {
        $existingZip = Find-ExistingFile $name ".zip"
        if ($existingZip) {
            Copy-Item -LiteralPath $existingZip -Destination $zipPath -Force
            Log "COPY existing zip $existingZip -> $zipPath"
        } else {
            Download-File "$OpenTTBase/$name.zip" $zipPath
        }
    }

    $annAlready = Find-AnnotationDir $name
    if (!$annAlready) {
        Expand-Zip $zipPath $ExtractedOpenTT
    }

    if (!(Test-Path $mp4Path)) {
        $existingMp4 = Find-ExistingFile $name ".mp4"
        if ($existingMp4) {
            Copy-Item -LiteralPath $existingMp4 -Destination $mp4Path -Force
            Log "COPY existing mp4 $existingMp4 -> $mp4Path"
        } else {
            Download-File "$OpenTTBase/$name.mp4" $mp4Path
        }
    }
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
        Log "OK annotations $name <- $annSrc"
    } else {
        Log "MISSING annotations $name"
    }

    $vidSrc = Join-Path $RawOpenTT "$name.mp4"
    if (Test-Path $vidSrc) {
        Copy-Item -LiteralPath $vidSrc -Destination $vidDst -Force
        Log "OK video $name"
    } else {
        Log "MISSING video $name"
    }
}

foreach ($name in $TrainNames) { Sync-One $name "training" }
foreach ($name in $TestNames)  { Sync-One $name "test" }

$inventoryPath = Join-Path $Dataset "_ttnet_dataset_inventory.csv"
Get-ChildItem -Path $Dataset -Recurse -File |
    Select-Object FullName, Length |
    Export-Csv -NoTypeInformation -Encoding UTF8 -Path $inventoryPath

Write-Host ""
Write-Host "=== CHECK FINAL ==="

Write-Host ""
Write-Host "Training videos:"
Get-ChildItem "$Dataset\training\videos" -Filter *.mp4 -ErrorAction SilentlyContinue | Select-Object Name,Length

Write-Host ""
Write-Host "Test videos:"
Get-ChildItem "$Dataset\test\videos" -Filter *.mp4 -ErrorAction SilentlyContinue | Select-Object Name,Length

Write-Host ""
Write-Host "Annotation files:"
Get-ChildItem "$Dataset\*\annotations\*" -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -in @("ball_markup.json","events_markup.json") } |
    Select-Object FullName,Length

Write-Host ""
Write-Host "Inventory: $inventoryPath"
Write-Host "Log: $Log"
Log "DONE 006C"

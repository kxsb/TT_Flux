param(
    [switch]$FullOpenTTGames,
    [switch]$T3SetFull,
    [switch]$Kaggle,
    [switch]$Roboflow,
    [switch]$TT3DRaw,
    [switch]$BlurBallData,
    [switch]$BlurBallWeights,
    [switch]$NoExtract
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {}

$Root      = (Get-Location).Path
$Base      = Join-Path $Root "data_external"
$Raw       = Join-Path $Base "raw"
$Extracted = Join-Path $Base "extracted"
$Docs      = Join-Path $Base "docs"
$Code      = Join-Path $Base "code"
$Index     = Join-Path $Base "index"
$Log       = Join-Path $Base "_download_log.txt"
$Manifest  = Join-Path $Base "dataset_sources_manifest.csv"

$Dirs = @(
    $Base, $Raw, $Extracted, $Docs, $Code, $Index,
    (Join-Path $Raw "openttgames"),
    (Join-Path $Extracted "openttgames"),
    (Join-Path $Raw "zenodo_t3set"),
    (Join-Path $Extracted "zenodo_t3set"),
    (Join-Path $Raw "kaggle_ball_position"),
    (Join-Path $Raw "kaggle_ttnet"),
    (Join-Path $Raw "roboflow_tabletennis"),
    (Join-Path $Raw "tt3d_raw"),
    (Join-Path $Raw "blurball_dataset"),
    (Join-Path $Raw "blurball_weights"),
    (Join-Path $Raw "dtu_table_tennis_data"),
    (Join-Path $Docs "papers"),
    (Join-Path $Docs "webpages")
)

foreach ($d in $Dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

if (Test-Path $Log) {
    Remove-Item $Log -Force
}

$Rows = New-Object System.Collections.Generic.List[Object]

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $Message"
    $line | Tee-Object -FilePath $Log -Append
}

function Add-ManifestRow {
    param(
        [string]$Slug,
        [string]$Category,
        [string]$Status,
        [string]$LocalPath,
        [string]$SourceUrl,
        [string]$Note
    )
    $script:Rows.Add([pscustomobject]@{
        slug       = $Slug
        category   = $Category
        status     = $Status
        local_path = $LocalPath
        source_url = $SourceUrl
        note       = $Note
    }) | Out-Null
}

function Has-Cmd {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Download-File {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$OutFile
    )

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null

    if ((Test-Path $OutFile) -and ((Get-Item $OutFile).Length -gt 0)) {
        Write-Log "SKIP exists: $OutFile"
        return
    }

    Write-Log "DOWNLOAD: $Url -> $OutFile"

    if (Has-Cmd "aria2c") {
        & aria2c -c -x 8 -s 8 --console-log-level=warn -d (Split-Path -Parent $OutFile) -o (Split-Path -Leaf $OutFile) $Url
        if ($LASTEXITCODE -ne 0) { throw "aria2c failed for $Url" }
        return
    }

    if (Has-Cmd "curl.exe") {
        & curl.exe -L --fail --retry 5 --retry-delay 3 -C - -o "$OutFile" "$Url"
        if ($LASTEXITCODE -ne 0) {
            Write-Log "curl resume failed, retry without resume: $Url"
            & curl.exe -L --fail --retry 5 --retry-delay 3 -o "$OutFile" "$Url"
            if ($LASTEXITCODE -ne 0) { throw "curl failed for $Url" }
        }
        return
    }

    Invoke-WebRequest -Uri $Url -OutFile $OutFile
}

function Expand-ZipSafe {
    param(
        [string]$ZipPath,
        [string]$Dest
    )

    if ($NoExtract) {
        Write-Log "NOEXTRACT: $ZipPath"
        return
    }

    if (!(Test-Path $ZipPath)) {
        Write-Log "SKIP extract missing: $ZipPath"
        return
    }

    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    Write-Log "EXTRACT: $ZipPath -> $Dest"

    try {
        Expand-Archive -LiteralPath $ZipPath -DestinationPath $Dest -Force
    } catch {
        if (Has-Cmd "7z") {
            & 7z x "$ZipPath" "-o$Dest" -y
            if ($LASTEXITCODE -ne 0) { throw "7z failed for $ZipPath" }
        } else {
            Write-Log "WARN extract failed and 7z missing: $ZipPath"
        }
    }
}

function Ensure-GitRepo {
    param(
        [string]$RepoUrl,
        [string]$Dest,
        [string]$FallbackZipUrl
    )

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dest) | Out-Null

    if (Has-Cmd "git") {
        if (Test-Path (Join-Path $Dest ".git")) {
            Write-Log "GIT pull: $Dest"
            & git -C "$Dest" pull --ff-only
            if ($LASTEXITCODE -ne 0) { Write-Log "WARN git pull failed: $RepoUrl" }
        } else {
            Write-Log "GIT clone: $RepoUrl -> $Dest"
            & git clone --depth 1 "$RepoUrl" "$Dest"
            if ($LASTEXITCODE -ne 0) { Write-Log "WARN git clone failed: $RepoUrl" }
        }
        return
    }

    if ($FallbackZipUrl) {
        $zipPath = "$Dest.zip"
        Download-File $FallbackZipUrl $zipPath
        Expand-ZipSafe $zipPath $Dest
    } else {
        Write-Log "SKIP git missing and no fallback zip: $RepoUrl"
    }
}

Write-Log "TTFlux external dataset aggregation start"
Write-Log "Root = $Root"

# -------------------------------------------------------------------
# 1) OpenTTGames : annotations légères par défaut ; vidéos avec -FullOpenTTGames
# -------------------------------------------------------------------

$OpenTTBase = "https://lab.osai.ai/datasets/openttgames/data"
$OpenTTMarkups = @(
    "game_1.zip","game_2.zip","game_3.zip","game_4.zip","game_5.zip",
    "test_1.zip","test_2.zip","test_3.zip","test_4.zip","test_5.zip","test_6.zip","test_7.zip"
)

foreach ($name in $OpenTTMarkups) {
    $url = "$OpenTTBase/$name"
    $out = Join-Path (Join-Path $Raw "openttgames") $name
    Download-File $url $out
    Expand-ZipSafe $out (Join-Path $Extracted "openttgames")
}

Add-ManifestRow `
    -Slug "openttgames_annotations" `
    -Category "dataset/ball_events_segmentation" `
    -Status "downloaded_default" `
    -LocalPath "data_external/raw/openttgames + data_external/extracted/openttgames" `
    -SourceUrl "https://lab.osai.ai/" `
    -Note "Annotations, ball coords, events, masks. Videos only if -FullOpenTTGames."

if ($FullOpenTTGames) {
    $OpenTTVideos = @(
        "game_1.mp4","game_2.mp4","game_3.mp4","game_4.mp4","game_5.mp4",
        "test_1.mp4","test_2.mp4","test_3.mp4","test_4.mp4","test_5.mp4","test_6.mp4","test_7.mp4"
    )

    foreach ($name in $OpenTTVideos) {
        $url = "$OpenTTBase/$name"
        $out = Join-Path (Join-Path $Raw "openttgames") $name
        Download-File $url $out
    }

    Add-ManifestRow `
        -Slug "openttgames_videos" `
        -Category "dataset/full_hd_120fps_video" `
        -Status "downloaded_when_FullOpenTTGames" `
        -LocalPath "data_external/raw/openttgames/*.mp4" `
        -SourceUrl "https://lab.osai.ai/" `
        -Note "Gros volume. Base TTNet/OpenTTGames complète."
}

# -------------------------------------------------------------------
# 2) TT3D code + evaluation dataset GitHub
# -------------------------------------------------------------------

Ensure-GitRepo `
    -RepoUrl "https://github.com/cogsys-tuebingen/tt3d.git" `
    -Dest (Join-Path $Code "tt3d") `
    -FallbackZipUrl "https://github.com/cogsys-tuebingen/tt3d/archive/refs/heads/main.zip"

Add-ManifestRow `
    -Slug "tt3d_code_and_eval" `
    -Category "techno/3d_reconstruction/camera_calibration" `
    -Status "cloned_or_downloaded" `
    -LocalPath "data_external/code/tt3d" `
    -SourceUrl "https://github.com/cogsys-tuebingen/tt3d" `
    -Note "Code + data/evaluation inclus dans le repo."

if ($TT3DRaw) {
    $tt3dRawZip = Join-Path (Join-Path $Raw "tt3d_raw") "tt3d_raw_dataset_nextcloud.zip"
    Download-File "https://cloud.cs.uni-tuebingen.de/index.php/s/SCKq85JZEmKoC6J/download" $tt3dRawZip
    Expand-ZipSafe $tt3dRawZip (Join-Path $Extracted "tt3d_raw")

    Add-ManifestRow `
        -Slug "tt3d_raw_dataset" `
        -Category "dataset/raw_3d_reconstruction_video" `
        -Status "downloaded_when_TT3DRaw" `
        -LocalPath "data_external/raw/tt3d_raw" `
        -SourceUrl "https://cogsys-tuebingen.github.io/tt3d/" `
        -Note "Raw dataset Nextcloud. Vérifier le zip si le serveur renvoie une page HTML."
}

# -------------------------------------------------------------------
# 3) BlurBall code + dataset/weights optionnels
# -------------------------------------------------------------------

Ensure-GitRepo `
    -RepoUrl "https://github.com/cogsys-tuebingen/blurball.git" `
    -Dest (Join-Path $Code "blurball") `
    -FallbackZipUrl "https://github.com/cogsys-tuebingen/blurball/archive/refs/heads/main.zip"

Add-ManifestRow `
    -Slug "blurball_code" `
    -Category "techno/ball_tracking/blur_estimation" `
    -Status "cloned_or_downloaded" `
    -LocalPath "data_external/code/blurball" `
    -SourceUrl "https://github.com/cogsys-tuebingen/blurball" `
    -Note "Très utile pour vidéos compressées/floues."

if ($BlurBallData) {
    $bbZip = Join-Path (Join-Path $Raw "blurball_dataset") "blurball_dataset_nextcloud.zip"
    Download-File "https://cloud.cs.uni-tuebingen.de/index.php/s/C3pJEPKWQAkono7/download" $bbZip
    Expand-ZipSafe $bbZip (Join-Path $Extracted "blurball_dataset")

    Add-ManifestRow `
        -Slug "blurball_dataset" `
        -Category "dataset/ball_blur_detection" `
        -Status "downloaded_when_BlurBallData" `
        -LocalPath "data_external/raw/blurball_dataset" `
        -SourceUrl "https://github.com/cogsys-tuebingen/blurball" `
        -Note "Dataset Nextcloud. Vérifier le zip si le serveur renvoie une page HTML."
}

if ($BlurBallWeights) {
    $bbwZip = Join-Path (Join-Path $Raw "blurball_weights") "blurball_models_nextcloud.zip"
    Download-File "https://cloud.cs.uni-tuebingen.de/index.php/s/6Z8TpM3sXRKHzGC/download" $bbwZip
    Expand-ZipSafe $bbwZip (Join-Path $Extracted "blurball_weights")

    Add-ManifestRow `
        -Slug "blurball_weights" `
        -Category "weights/ball_tracking" `
        -Status "downloaded_when_BlurBallWeights" `
        -LocalPath "data_external/raw/blurball_weights" `
        -SourceUrl "https://github.com/cogsys-tuebingen/blurball" `
        -Note "Poids pré-entraînés BlurBall/WASB/TrackNet/etc."
}

# -------------------------------------------------------------------
# 4) SportsVideo / Centrale Lyon
# -------------------------------------------------------------------

Ensure-GitRepo `
    -RepoUrl "https://github.com/centralelyon/sportsvideo.git" `
    -Dest (Join-Path $Code "sportsvideo") `
    -FallbackZipUrl "https://github.com/centralelyon/sportsvideo/archive/refs/heads/main.zip"

Add-ManifestRow `
    -Slug "sportsvideo_centrale_lyon" `
    -Category "dataset/event_detection/strokes/audio/table_homography/score" `
    -Status "cloned_or_downloaded_if_public" `
    -LocalPath "data_external/code/sportsvideo" `
    -SourceUrl "https://github.com/centralelyon/sportsvideo" `
    -Note "MediaEval/SportsVideo. Utile pour strokes, table projection, son, score."

# -------------------------------------------------------------------
# 5) DTU / Extended OpenTTGames source repo
# -------------------------------------------------------------------

Ensure-GitRepo `
    -RepoUrl "https://gitlab.compute.dtu.dk/emilh/table_tennis_data.git" `
    -Dest (Join-Path $Raw "dtu_table_tennis_data") `
    -FallbackZipUrl ""

Add-ManifestRow `
    -Slug "dtu_extended_openttgames" `
    -Category "dataset/extended_stroke_annotations" `
    -Status "clone_attempted" `
    -LocalPath "data_external/raw/dtu_table_tennis_data" `
    -SourceUrl "https://gitlab.compute.dtu.dk/emilh/table_tennis_data" `
    -Note "GitLab public mais peut échouer selon droits/réseau."

# -------------------------------------------------------------------
# 6) Zenodo T3Set : README par défaut ; dataset complet avec -T3SetFull
# -------------------------------------------------------------------

$T3Dir = Join-Path $Raw "zenodo_t3set"
Download-File "https://zenodo.org/records/15516144/files/README.pdf?download=1" (Join-Path $T3Dir "README.pdf")

Add-ManifestRow `
    -Slug "t3set_readme" `
    -Category "dataset/multimodal_training/video_sensor_text" `
    -Status "readme_downloaded_default" `
    -LocalPath "data_external/raw/zenodo_t3set/README.pdf" `
    -SourceUrl "https://zenodo.org/records/15516144" `
    -Note "Dataset complet optionnel avec -T3SetFull."

if ($T3SetFull) {
    foreach ($i in 1..38) {
        $part = "z{0:D2}" -f $i
        $name = "T3Set_full_data.$part"
        Download-File "https://zenodo.org/records/15516144/files/$name?download=1" (Join-Path $T3Dir $name)
    }

    Download-File "https://zenodo.org/records/15516144/files/T3Set_full_data.zip?download=1" (Join-Path $T3Dir "T3Set_full_data.zip")

    if (!$NoExtract) {
        if (Has-Cmd "7z") {
            New-Item -ItemType Directory -Force -Path (Join-Path $Extracted "zenodo_t3set") | Out-Null
            & 7z x (Join-Path $T3Dir "T3Set_full_data.zip") "-o$(Join-Path $Extracted "zenodo_t3set")" -y
            if ($LASTEXITCODE -ne 0) { Write-Log "WARN 7z extraction failed for T3Set" }
        } else {
            Write-Log "WARN T3Set is a multipart zip. Install 7-Zip and run extraction from T3Set_full_data.zip."
        }
    }

    Add-ManifestRow `
        -Slug "t3set_full" `
        -Category "dataset/multimodal_training/video_sensor_text" `
        -Status "downloaded_when_T3SetFull" `
        -LocalPath "data_external/raw/zenodo_t3set" `
        -SourceUrl "https://zenodo.org/records/15516144" `
        -Note "Gros volume multipart. Extraction fiable avec 7z."
}

# -------------------------------------------------------------------
# 7) Kaggle optionnel : nécessite kaggle.json
# -------------------------------------------------------------------

if ($Kaggle) {
    if (Has-Cmd "kaggle") {
        $kg1 = Join-Path $Raw "kaggle_ball_position"
        $kg2 = Join-Path $Raw "kaggle_ttnet"

        & kaggle datasets download -d ketzoomer/table-tennis-ball-position-detection-dataset -p "$kg1" --unzip
        if ($LASTEXITCODE -ne 0) { Write-Log "WARN Kaggle ketzoomer failed" }

        & kaggle datasets download -d anshulmehtakaggl/table-tennis-games-dataset-ttnet -p "$kg2" --unzip
        if ($LASTEXITCODE -ne 0) { Write-Log "WARN Kaggle TTNet failed" }

        Add-ManifestRow `
            -Slug "kaggle_ball_position" `
            -Category "dataset/ball_position_detection" `
            -Status "download_attempted_with_kaggle_cli" `
            -LocalPath "data_external/raw/kaggle_ball_position" `
            -SourceUrl "https://www.kaggle.com/datasets/ketzoomer/table-tennis-ball-position-detection-dataset" `
            -Note "Nécessite compte Kaggle + kaggle.json."

        Add-ManifestRow `
            -Slug "kaggle_ttnet_games" `
            -Category "dataset/ttnet_games" `
            -Status "download_attempted_with_kaggle_cli" `
            -LocalPath "data_external/raw/kaggle_ttnet" `
            -SourceUrl "https://www.kaggle.com/datasets/anshulmehtakaggl/table-tennis-games-dataset-ttnet" `
            -Note "Nécessite compte Kaggle + kaggle.json."
    } else {
        Write-Log "SKIP Kaggle: installe le CLI avec: py -m pip install kaggle"
    }
}

# -------------------------------------------------------------------
# 8) Roboflow optionnel : nécessite ROBOFLOW_API_KEY
# -------------------------------------------------------------------

if ($Roboflow) {
    $rfOut = Join-Path $Raw "roboflow_tabletennis"

    if (!$env:ROBOFLOW_API_KEY) {
        Write-Log "SKIP Roboflow: variable ROBOFLOW_API_KEY absente."
    } else {
        $Py = $null
        if (Has-Cmd "py") { $Py = "py" }
        elseif (Has-Cmd "python") { $Py = "python" }

        if (!$Py) {
            Write-Log "SKIP Roboflow: Python introuvable."
        } else {
            $rfScript = Join-Path $Base "_download_roboflow_tabletennis.py"
            Set-Content -Encoding UTF8 -Path $rfScript -Value @"
import os
import sys
from roboflow import Roboflow

out = sys.argv[1]
api_key = os.environ["ROBOFLOW_API_KEY"]

rf = Roboflow(api_key=api_key)
project = rf.workspace("yolo-class-delww").project("tabletennis-nbcdc")
dataset = project.version(2).download("yolov8", location=out)
print("Roboflow downloaded to:", dataset.location)
"@

            & $Py -m pip show roboflow | Out-Null
            if ($LASTEXITCODE -ne 0) {
                & $Py -m pip install roboflow
            }

            & $Py $rfScript "$rfOut"
            if ($LASTEXITCODE -ne 0) { Write-Log "WARN Roboflow download failed" }

            Add-ManifestRow `
                -Slug "roboflow_tabletennis_v2_yolov8" `
                -Category "dataset/object_detection/yolo" `
                -Status "download_attempted_with_api_key" `
                -LocalPath "data_external/raw/roboflow_tabletennis" `
                -SourceUrl "https://universe.roboflow.com/yolo-class-delww/tabletennis-nbcdc/dataset/2" `
                -Note "Nécessite ROBOFLOW_API_KEY."
        }
    }
}

# -------------------------------------------------------------------
# 9) Docs / papiers / pages utiles
# -------------------------------------------------------------------

Download-File "https://perso.liris.cnrs.fr/marc.plantevit/ENS/DM/%5BSciences2024%5D-tfe-tennis-table.pdf" `
    (Join-Path (Join-Path $Docs "papers") "sciences2024_tfe_tennis_table.pdf")

Download-File "https://arxiv.org/pdf/2512.19327" `
    (Join-Path (Join-Path $Docs "papers") "extended_openttgames_2512_19327.pdf")

Download-File "https://arxiv.org/pdf/2605.01234" `
    (Join-Path (Join-Path $Docs "papers") "tt4d_2605_01234.pdf")

Download-File "https://ceur-ws.org/Vol-3658/paper3.pdf" `
    (Join-Path (Join-Path $Docs "papers") "sportsvideo_mediaeval2023.pdf")

Download-File "https://data.scorenetwork.org/table_tennis/table_tennis_sept2022.html" `
    (Join-Path (Join-Path $Docs "webpages") "scorenetwork_setka_table_tennis_sept2022.html")

Download-File "https://hyper.ai/fr/datasets/21496" `
    (Join-Path (Join-Path $Docs "webpages") "hyper_ai_dataset_21496.html")

Add-ManifestRow `
    -Slug "docs_and_papers" `
    -Category "docs/research/practical_guides" `
    -Status "downloaded" `
    -LocalPath "data_external/docs" `
    -SourceUrl "multiple" `
    -Note "Papiers, pages de référence, guides."

# -------------------------------------------------------------------
# 10) README + index fichiers
# -------------------------------------------------------------------

$Readme = @"
# TTFlux — external table tennis datasets

Generated by scripts/download_external_tt_datasets_006A.ps1

## Layout

- raw/: archives, vidéos, repos de données brutes.
- extracted/: archives décompressées.
- code/: repos Git utiles.
- docs/: papiers, pages HTML, notes.
- index/: inventaire local.
- dataset_sources_manifest.csv: manifest logique des sources.

## Recommended first usage

1. OpenTTGames annotations:
   data_external/extracted/openttgames

2. TT3D:
   data_external/code/tt3d

3. BlurBall:
   data_external/code/blurball

4. SportsVideo:
   data_external/code/sportsvideo

## Heavy options

- -FullOpenTTGames : télécharge aussi les MP4 OpenTTGames.
- -T3SetFull : télécharge l'archive multipart Zenodo T3Set.
- -TT3DRaw : télécharge le raw dataset TT3D Nextcloud.
- -BlurBallData : télécharge le dataset BlurBall Nextcloud.
- -BlurBallWeights : télécharge les poids BlurBall Nextcloud.
- -Kaggle : nécessite Kaggle CLI + kaggle.json.
- -Roboflow : nécessite ROBOFLOW_API_KEY.
"@

Set-Content -Encoding UTF8 -Path (Join-Path $Base "README_EXTERNAL_DATASETS.md") -Value $Readme

Get-ChildItem -Path $Base -Recurse -File |
    Select-Object FullName, Length, LastWriteTime |
    Export-Csv -Path (Join-Path $Index "file_inventory.csv") -NoTypeInformation -Encoding UTF8

$Rows | Export-Csv -Path $Manifest -NoTypeInformation -Encoding UTF8

Write-Log "Manifest: $Manifest"
Write-Log "Inventory: $(Join-Path $Index "file_inventory.csv")"
Write-Log "Done."

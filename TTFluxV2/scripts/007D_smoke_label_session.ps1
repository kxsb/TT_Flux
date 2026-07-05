$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python -m compileall -q ttflux app

& $Python - <<'PY'
from ttflux.labels.session import list_sessions
print("OK labels module")
print("sessions=", len(list_sessions()))
PY

<#
.SYNOPSIS
    Launch the full Kiln stack (FastAPI backend + Electron/Vite UI) with one command.

.DESCRIPTION
    Starts the unified Kiln backend (uvicorn on kiln_server:app) and the UI in
    separate console windows so each keeps its own live logs. Closing a window
    stops that process.

.PARAMETER Port
    Backend port. Defaults to 8420.

.PARAMETER Browser
    Run the UI as a plain Vite dev server (open http://localhost:5173 yourself)
    instead of launching the Electron desktop window.

.PARAMETER NoUi
    Start only the backend.

.PARAMETER InstallDeps
    Run "npm install" in ui/ before launching if dependencies look missing.

.EXAMPLE
    .\start-kiln.ps1
    Start backend + Electron UI.

.EXAMPLE
    .\start-kiln.ps1 -Browser
    Start backend + Vite dev server (browser-based UI).
#>
[CmdletBinding()]
param(
    [int]$Port = 8420,
    [switch]$Browser,
    [switch]$NoUi,
    [switch]$InstallDeps
)

$ErrorActionPreference = 'Stop'

# Keep the window open on any error so failures are readable when double-clicked.
trap {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
}

# Resolve repo root from this script's location so it works from anywhere.
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$UiDir = Join-Path $Root 'ui'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "    $msg" -ForegroundColor Yellow }

Write-Step "Kiln launcher (root: $Root)"

# --- Preflight: required tools -------------------------------------------------
function Test-Command($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

if (-not (Test-Command 'python')) {
    throw "python not found on PATH. Install Python 3.10+ and retry."
}
if (-not $NoUi) {
    if (-not (Test-Command 'npm')) {
        throw "npm not found on PATH. Install Node.js and retry (or use -NoUi)."
    }
}

# --- Preflight: backend imports ------------------------------------------------
Write-Step "Checking backend imports"
Push-Location $Root
try {
    python -c "import chonk, fastapi, uvicorn" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warn2 "Backend dependencies missing. Installing editable package..."
        pip install -e ".[dev]"
        if ($LASTEXITCODE -ne 0) { throw "pip install failed." }
    }
    Write-Ok "Backend imports OK"
}
finally {
    Pop-Location
}

# --- Preflight: UI dependencies ------------------------------------------------
if (-not $NoUi) {
    $nodeModules = Join-Path $UiDir 'node_modules'
    if ($InstallDeps -or -not (Test-Path $nodeModules)) {
        Write-Step "Installing UI dependencies (npm install)"
        Push-Location $UiDir
        try {
            npm install
            if ($LASTEXITCODE -ne 0) { throw "npm install failed." }
        }
        finally {
            Pop-Location
        }
    }
    Write-Ok "UI dependencies present"
}

# --- Preflight: free stale Kiln ports so re-launches start clean ---------------
function Clear-Port($p) {
    try {
        Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique |
            ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    }
    catch { }
}
Write-Step "Freeing stale Kiln ports (clears leftover processes from a prior run)"
Clear-Port $Port
if (-not $NoUi) { Clear-Port 5173 }
Start-Sleep -Milliseconds 500
Write-Ok "Ports $Port$(if (-not $NoUi) { ' and 5173' }) clear"

# --- Launch backend ------------------------------------------------------------
Write-Step "Starting backend (uvicorn) on port $Port"
$backendCmd = "Set-Location '$Root'; " +
              "Write-Host 'Kiln backend -- http://localhost:$Port/docs' -ForegroundColor Cyan; " +
              "python -m uvicorn kiln_server:app --reload --port $Port"
$backend = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList '-NoExit', '-Command', $backendCmd `
    -PassThru
Write-Ok "Backend window started (PID $($backend.Id))"

# --- Wait for backend health ---------------------------------------------------
Write-Step "Waiting for backend health (first start can take ~60s)"
$healthUrl = "http://localhost:$Port/api/health"
$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        $ready = $true
        $loaded = ($resp.tools.PSObject.Properties |
                   Where-Object { $_.Value.loaded } |
                   ForEach-Object { $_.Name }) -join ', '
        Write-Ok "Backend status: $($resp.status). Tools loaded: $loaded"
        break
    }
    catch {
        # not up yet; keep polling
    }
}
if (-not $ready) {
    Write-Warn2 "Backend not healthy yet. It may still be loading -- check its window and http://localhost:$Port/api/health."
}

# --- Launch UI -----------------------------------------------------------------
if (-not $NoUi) {
    $uiScript = if ($Browser) { 'dev' } else { 'electron:dev' }
    Write-Step "Starting UI (npm run $uiScript)"
    $uiCmd = "Set-Location '$UiDir'; " +
             "Write-Host 'Kiln UI -- http://localhost:5173' -ForegroundColor Cyan; " +
             "npm run $uiScript"
    $ui = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList '-NoExit', '-Command', $uiCmd `
        -PassThru
    Write-Ok "UI window started (PID $($ui.Id))"
}

Write-Host ""
Write-Step "Kiln is starting up"
Write-Host "    Backend : http://localhost:$Port/api/health" -ForegroundColor Green
Write-Host "    API docs: http://localhost:$Port/docs" -ForegroundColor Green
if (-not $NoUi) {
    if ($Browser) {
        Write-Host "    UI      : http://localhost:5173  (open in your browser)" -ForegroundColor Green
    }
    else {
        Write-Host "    UI      : Electron window (opening shortly)" -ForegroundColor Green
    }
}
Write-Host ""
Write-Host "Each process runs in its own window. Close a window to stop that process." -ForegroundColor DarkGray

# ClarityAI - one-shot launcher (single process)
# -----------------------------------------------
# What this does:
#   1. Kills every leftover node and python process so there are no zombies
#   2. Confirms the api.ts fix is in place (re-applies it if not)
#   3. Builds the frontend so uvicorn can serve it
#   4. Starts ONE uvicorn process that serves BOTH the API and the UI
#   5. Opens http://127.0.0.1:8000 in your default browser
#
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1
# Or right-click -> Run with PowerShell.

$ErrorActionPreference = 'Stop'

# Project root = the folder this script lives in.
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  ClarityAI single-process launcher" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# ---- 1. Kill stale dev-server processes ----
Write-Host "[1/5] Stopping any leftover node / vite / uvicorn processes..." -ForegroundColor Yellow
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
# Only stop python processes that look like uvicorn workers - be conservative.
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*uvicorn*' -or $_.CommandLine -like '*app.main*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1

# ---- 2. Verify the api.ts fix is in place ----
Write-Host "[2/5] Verifying frontend\src\lib\api.ts fix..." -ForegroundColor Yellow
$apiFile = Join-Path $Root 'frontend\src\lib\api.ts'
if (-not (Test-Path $apiFile)) {
    Write-Host "ERROR: $apiFile does not exist." -ForegroundColor Red
    exit 1
}
$line12 = (Get-Content $apiFile -TotalCount 16) | Select-Object -Last 5 | Out-String
if ($line12 -notmatch 'const API_BASE\s*=') {
    Write-Host "  api.ts looks unpatched - applying fix..." -ForegroundColor Yellow
    $original = Get-Content $apiFile -Raw
    $fixed = $original -replace `
        "const API_BASE_URL = import\.meta\.env\.VITE_API_BASE_URL \|\| 'http://127\.0\.0\.1:8000'", `
        "const API_BASE = ((import.meta.env.VITE_API_BASE_URL || '').replace(/\/`$/, '')) + '/api'"
    if ($fixed -eq $original) {
        Write-Host "  ERROR: could not find the line to patch in api.ts." -ForegroundColor Red
        Write-Host "  Open frontend\src\lib\api.ts and make sure line 12 starts with: const API_BASE =" -ForegroundColor Red
        exit 1
    }
    $fixed | Set-Content -Path $apiFile -Encoding UTF8
    Write-Host "  Patched." -ForegroundColor Green
} else {
    Write-Host "  Already patched. Good." -ForegroundColor Green
}

# ---- 3. Build the frontend ----
Write-Host "[3/5] Building the frontend (npm run build)..." -ForegroundColor Yellow
Push-Location (Join-Path $Root 'frontend')
if (-not (Test-Path 'node_modules')) {
    Write-Host "  node_modules missing, running npm install (this can take a couple minutes)..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Host "npm install failed." -ForegroundColor Red; exit 1 }
}
# Wipe any stale build cache and old dist - we want a guaranteed-fresh build.
Remove-Item -Recurse -Force 'node_modules\.vite' -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force 'dist' -ErrorAction SilentlyContinue
npm run build
if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Host "npm run build failed." -ForegroundColor Red; exit 1 }
Pop-Location
Write-Host "  Build done." -ForegroundColor Green

# ---- 4. Make sure backend deps are installed ----
Write-Host "[4/5] Checking backend Python dependencies..." -ForegroundColor Yellow
$venvPython = Join-Path $Root 'backend\.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host "  Creating Python venv at backend\.venv..." -ForegroundColor Yellow
    Push-Location (Join-Path $Root 'backend')
    py -m venv .venv 2>$null
    if (-not (Test-Path '.venv\Scripts\python.exe')) { python -m venv .venv }
    Pop-Location
}
& $venvPython -c "import fastapi" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing backend dependencies..." -ForegroundColor Yellow
    & $venvPython -m pip install --upgrade pip --quiet
    & $venvPython -m pip install -r (Join-Path $Root 'backend\requirements.txt')
    if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed." -ForegroundColor Red; exit 1 }
}
Write-Host "  Backend ready." -ForegroundColor Green

# ---- 5. Start uvicorn and open the browser ----
Write-Host "[5/5] Starting ClarityAI on http://127.0.0.1:8000 ..." -ForegroundColor Yellow
Write-Host ""
Write-Host "    Open this URL: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "    (Press Ctrl+C in this window to stop the server)" -ForegroundColor DarkGray
Write-Host ""
Start-Sleep -Seconds 1
Start-Process "http://127.0.0.1:8000"

Push-Location (Join-Path $Root 'backend')
& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Pop-Location
@echo off
REM ClarityAI — one-click Windows starter
REM
REM Prerequisites (install once):
REM   - Python 3.11 or 3.12   https://www.python.org/downloads/
REM   - Node.js 20 LTS        https://nodejs.org/
REM
REM What this script does:
REM   1) Creates a Python venv in backend\.venv if missing
REM   2) Installs backend dependencies if missing
REM   3) Installs frontend dependencies if missing
REM   4) Opens TWO new terminal windows: one for backend, one for frontend
REM   5) Opens the app in your browser at http://localhost:5173

setlocal
cd /d "%~dp0"

echo.
echo ===============================================
echo   ClarityAI - first-run setup and launch
echo ===============================================
echo.

REM ---- Sanity check: backend\.env exists ----
if not exist "backend\.env" (
  echo No backend\.env found. Copying from .env.example...
  copy /y "backend\.env.example" "backend\.env" >nul
  echo Edit backend\.env now and paste your Groq API key on the LLM_API_KEY line.
  echo Then run start.bat again.
  notepad backend\.env
  pause
  exit /b 0
)

REM ---- Warn if API key is still placeholder ----
findstr /c:"PASTE_YOUR_GROQ_KEY_HERE" "backend\.env" >nul
if %errorlevel%==0 (
  echo.
  echo WARNING: backend\.env still has the placeholder key.
  echo Open backend\.env and paste your Groq API key on the LLM_API_KEY line.
  echo The app will run, but answers will fall back to evidence-only mode until you add the key.
  echo.
  pause
)

REM ---- Backend venv ----
if not exist "backend\.venv\Scripts\python.exe" (
  echo [1/3] Creating Python virtual environment...
  pushd backend
  py -m venv .venv 2>nul || python -m venv .venv
  if errorlevel 1 (
    echo Failed to create venv. Is Python installed and on PATH?
    popd
    pause
    exit /b 1
  )
  popd
)

REM ---- Backend deps ----
if not exist "backend\.venv\Lib\site-packages\fastapi" (
  echo [2/3] Installing backend dependencies (this can take a couple of minutes)...
  pushd backend
  .venv\Scripts\python.exe -m pip install --upgrade pip >nul
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Backend pip install failed. See the error above.
    popd
    pause
    exit /b 1
  )
  popd
)

REM ---- Frontend deps ----
if not exist "frontend\node_modules" (
  echo [3/3] Installing frontend dependencies...
  pushd frontend
  call npm install
  if errorlevel 1 (
    echo Frontend npm install failed. Is Node.js installed?
    popd
    pause
    exit /b 1
  )
  popd
)

echo.
echo Starting backend in a new window...
start "ClarityAI backend" cmd /k "cd /d %~dp0backend && .venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo Starting frontend in a new window...
start "ClarityAI frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo Waiting for the frontend to come up...
timeout /t 4 /nobreak >nul

echo Opening browser...
start "" "http://localhost:5173"

echo.
echo ===============================================
echo   ClarityAI is starting.
echo   Backend:  http://localhost:8000  (docs: /docs)
echo   Frontend: http://localhost:5173
echo
echo   Close the two new windows to stop the app.
echo ===============================================
echo.
endlocal

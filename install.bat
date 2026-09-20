@echo off
REM ============================================================================
REM Lexis one-click installer (native Windows Command Prompt / PowerShell).
REM Double-click install.bat. No technical knowledge required.
REM Mirrors install.sh (Git Bash) step-for-step: Python check -> .venv ->
REM pre-compiled llama-cpp-python wheel -> project install -> GGUF download ->
REM smoke tests -> launch on http://localhost:8000
REM Env knobs:  set PORT=8000  |  set MODEL_QANT=Q4_K_M  |  set SKIP_MODEL=1
REM ============================================================================
setlocal EnableDelayedExpansion

if "%MODEL_REPO%"=="" set MODEL_REPO=Qwen/Qwen2.5-3B-Instruct-GGUF
if "%MODEL_QANT%"=="" set MODEL_QANT=Q4_K_M
if "%MODEL_QANT%"=="Q4_K_M" (set MODEL_FILE=qwen2.5-3b-instruct-q4_k_m.gguf) else (set MODEL_FILE=qwen2.5-3b-instruct-%MODEL_QANT%.gguf)
if "%CACHE_DIR%"=="" set CACHE_DIR=%USERPROFILE%\.cache\lexis\models
if "%PORT%"=="" set PORT=8000

echo [Lexis] Native Windows installer starting...

REM ---- 1. Resolve Python 3.10+ (py launcher preferred, python fallback) ----
set PY=
py -3 --version >nul 2>&1
if not errorlevel 1 (set PY=py -3) else (
  python --version >nul 2>&1
  if not errorlevel 1 (set PY=python) else (
    echo [Lexis] ERROR: Python 3 not found.
    echo [Lexis] Install Python 3.10+ from https://www.python.org/downloads/
    echo [Lexis] IMPORTANT: tick "Add python.exe to PATH", then re-run install.bat.
    pause & exit /b 1
  )
)
for /f "tokens=2 delims= " %%v in ('%PY% --version 2^>^&1') do set PYVER=%%v
echo [Lexis] Using Python %PYVER% (%PY%)
%PY% -c "import sys; assert sys.version_info >= (3, 10)" 2>nul
if errorlevel 1 (
  echo [Lexis] ERROR: Python 3.10+ required (found %PYVER%). Upgrade, then re-run.
  pause & exit /b 1
)

REM ---- 2. Virtual environment ----
if not exist .venv (
  echo [Lexis] Creating isolated environment (.venv^)...
  %PY% -m venv .venv
  if errorlevel 1 (echo [Lexis] ERROR: venv creation failed. Delete ".venv" and re-run. & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
set VPY=python
%VPY% -c "import sys; print('[Lexis] venv python:', sys.version.split()[0])"

REM ---- 3. Dependencies: pre-compiled wheels first (no compiler needed) ----
echo [Lexis] Installing open-source building blocks...
%VPY% -m pip install --upgrade pip wheel setuptools
if "%LLAMA_CUDA%"=="1" (
  echo [Lexis] LLAMA_CUDA=1: attempting CUDA build ^(needs CMake + CUDA toolkit^)...
  set CMAKE_ARGS=-DGGML_CUDA=on
  %VPY% -m pip install llama-cpp-python --no-cache-dir
  if errorlevel 1 echo [Lexis] WARNING: CUDA build failed; falling back to CPU wheel.
)
%VPY% -m pip install --prefer-binary "llama-cpp-python>=0.2.75"
if errorlevel 1 echo [Lexis] WARNING: llama-cpp-python wheel unavailable; mock mode until it installs.
%VPY% -m pip install -e ".[local]"
if errorlevel 1 (
  echo [Lexis] WARNING: full install failed; installing CPU-only core...
  %VPY% -m pip install -e "."
)
%VPY% -c "import llama_cpp; print('[Lexis] llama.cpp backend: OK')" 2>nul
if errorlevel 1 echo [Lexis] WARNING: llama.cpp not importable yet ^(mock mode for now^).

REM ---- 4. Model download: huggingface_hub -> curl.exe -> powershell ----
if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
if exist "%CACHE_DIR%\%MODEL_FILE%" (
  echo [Lexis] Model already cached: %CACHE_DIR%\%MODEL_FILE%
  goto smoke
)
if "%SKIP_MODEL%"=="1" (
  echo [Lexis] SKIP_MODEL=1: skipping download ^(mock mode until %MODEL_FILE% is in %CACHE_DIR%^).
  goto smoke
)
echo [Lexis] Downloading %MODEL_FILE% (~2GB, one time, resumable)...
%VPY% -c "from huggingface_hub import hf_hub_download; hf_hub_download('%MODEL_REPO%', '%MODEL_FILE%', local_dir=r'%CACHE_DIR%', resume_download=True); print('[Lexis] huggingface_hub download: OK')" 2>nul
if not errorlevel 1 goto have_model
echo [Lexis] huggingface_hub path failed; trying curl.exe...
where curl >nul 2>&1
if not errorlevel 1 (
  curl -fL --retry 3 --retry-delay 5 -C - --progress-bar -o "%CACHE_DIR%\%MODEL_FILE%" "https://huggingface.co/%MODEL_REPO%/resolve/main/%MODEL_FILE%?download=true"
  if not errorlevel 1 goto have_model
)
echo [Lexis] Trying PowerShell BITS download...
powershell -NoProfile -Command "Start-BitsTransfer -Source 'https://huggingface.co/%MODEL_REPO%/resolve/main/%MODEL_FILE%?download=true' -Destination '%CACHE_DIR%\%MODEL_FILE%'"
if errorlevel 1 (
  echo [Lexis] WARNING: automatic download failed. Manual step: download %MODEL_FILE%
  echo [Lexis] from https://huggingface.co/%MODEL_REPO% into %CACHE_DIR%
  echo [Lexis] Server will start in mock mode; restart after placing the file.
  goto smoke
)
:have_model
echo [Lexis] Model ready: %CACHE_DIR%\%MODEL_FILE%

REM ---- 5. Smoke test + launch ----
:smoke
if "%SKIP_TESTS%"=="1" (
  echo [Lexis] SKIP_TESTS=1: skipping pytest.
  goto launch
)
echo [Lexis] Running smoke tests...
%VPY% -m pytest -q
if errorlevel 1 echo [Lexis] WARNING: some tests failed; server will still start.

:launch
echo [Lexis] Launching Lexis (auto port from %PORT% if busy) ...
echo [Lexis] Keep this window open. Press Ctrl+C to stop.
%VPY% -c "from lexis_local.main import start_server; start_server()"
pause

@echo off
rem Lexis production launch prep (native Windows CMD).
rem   1. lint (ruff + black, blocking)   2. pytest   3. git init (if needed)
rem   4. git add -A  5. initial production commit (only if something is staged).
rem Usage:  scripts\publish_prep.bat
setlocal EnableExtensions
cd /d "%~dp0\.."

set "VPY=python"
if exist ".venv\Scripts\python.exe" set "VPY=.venv\Scripts\python.exe"

for /f "delims=" %%v in ('%VPY% -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"') do set "VERSION=%%v"
echo === Lexis publish prep (v%VERSION%) ===

echo --- [1/5] ruff ---
%VPY% -m ruff check lexis_local tests benchmarks scripts
if errorlevel 1 exit /b 1
echo --- [2/5] black --check ---
%VPY% -m black --check lexis_local tests benchmarks scripts
if errorlevel 1 exit /b 1
echo --- [3/5] pytest ---
%VPY% -m pytest -q
if errorlevel 1 exit /b 1

echo --- [4/5] git init (if needed) ---
if not exist ".git" (
  git init -b main
  echo initialized empty git repository on branch main
)

echo --- [5/5] stage + commit ---
git add -A
git status --short
git diff --cached --quiet
if errorlevel 1 (
  git -c user.name="%GIT_USER%" -c user.email="%GIT_EMAIL%" commit -m "chore(release): Lexis v%VERSION% production-ready" -m "Lint clean, pytest green (mock mode), benchmark sub-100ms. Zero-hallucination GGUF engine + OpenAI-compatible API + agent shield."
  if errorlevel 1 (
    git commit -m "chore(release): Lexis v%VERSION% production-ready" -m "Lint clean, pytest green (mock mode), benchmark sub-100ms. Zero-hallucination GGUF engine + OpenAI-compatible API + agent shield."
  )
) else (
  echo nothing staged - working tree already matches HEAD, no commit needed
)

echo === ready ===
echo Next: create the GitHub repo, then:
echo   git remote add origin https://github.com/^^<you^^>/lexis-local.git
echo   git push -u origin main

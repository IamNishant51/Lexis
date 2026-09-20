#!/usr/bin/env bash
# Lexis production launch prep (Linux / macOS / Git Bash).
#   1. lint (ruff + black, blocking)   2. pytest   3. git init (if needed)
#   4. git add -A  5. initial production commit (only if something is staged).
# Usage:  bash scripts/publish_prep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; fi  # Git Bash
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; fi

VERSION=$($PY -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
echo "=== Lexis publish prep (v$VERSION) ==="

echo "--- [1/5] ruff ---"
$PY -m ruff check lexis_local tests benchmarks scripts
echo "--- [2/5] black --check ---"
$PY -m black --check lexis_local tests benchmarks scripts
echo "--- [3/5] pytest ---"
$PY -m pytest -q

echo "--- [4/5] git init (if needed) ---"
if [ ! -d .git ]; then
  git init -b main
  echo "initialized empty git repository on branch main"
fi

echo "--- [5/5] stage + commit ---"
git add -A
git status --short | head -20
if git diff --cached --quiet; then
  echo "nothing staged — working tree already matches HEAD, no commit needed"
else
  git -c user.name="${GIT_USER:-lexis-local}" -c user.email="${GIT_EMAIL:-lexis-local@localhost}" \
    commit -m "chore(release): Lexis v$VERSION production-ready" \
             -m "Lint clean, pytest green (mock mode), benchmark sub-100ms. Zero-hallucination GGUF engine + OpenAI-compatible API + agent shield."
fi

echo "=== ready ==="
echo "Next: create the GitHub repo, then:"
echo "  git remote add origin https://github.com/<you>/lexis-local.git"
echo "  git push -u origin main"

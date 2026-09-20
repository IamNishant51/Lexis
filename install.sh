#!/usr/bin/env bash
# Lexis one-click installer — Linux, macOS, AND Git Bash (MINGW64) on Windows.
# Usage:  bash install.sh        (double-click works in Git Bash via "Run")
# Env knobs: PORT=8000  MODEL_QANT=Q4_K_M  LLAMA_CUDA=1  SKIP_MODEL=1  SKIP_TESTS=1
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-3B-Instruct-GGUF}"
MODEL_QANT="${MODEL_QANT:-Q4_K_M}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-3b-instruct-${MODEL_QANT,,}.gguf}"
CACHE_DIR="${LEXIS_MODEL_DIR:-$HOME/.cache/lexis/models}"
PORT="${PORT:-8000}"

say() { printf '\033[1;32m[Lexis]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[Lexis]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[Lexis]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- OS detect
OS="linux"; FLAVOR=""
case "$(uname -s 2>/dev/null || echo Unknown)" in
  MINGW*|MSYS*|CYGWIN*) OS="windows"; FLAVOR="git-bash" ;;
  Darwin*) OS="macos" ;;
esac
say "Host detected: $OS${FLAVOR:+ ($FLAVOR)}"

# ------------------------------------------------- Python 3.10+ resolution
PY=""
for cand in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
# Git Bash usually lacks `python3` but has `py -3` / `python` from python.org.
if [ -z "$PY" ] && [ "$OS" = "windows" ]; then
  if command -v py >/dev/null 2>&1; then PY="py -3";
  elif command -v python >/dev/null 2>&1; then PY="python"; fi
fi
[ -n "$PY" ] || die "Python 3 not found. Install Python 3.10+ from https://www.python.org/downloads/ (Windows: tick 'Add python.exe to PATH'), then re-run."
PYVER=$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
say "Using Python $PYVER ($PY)"
$PY -c 'import sys; assert sys.version_info >= (3, 10)' \
  || die "Python 3.10+ required (found $PYVER). Upgrade, then re-run."
$PY -c 'import venv' 2>/dev/null \
  || die "Python 'venv' module missing. On Debian/Ubuntu: sudo apt install python3-venv. Then re-run."

# ------------------------------------------------- Virtual environment
if [ ! -d .venv ]; then
  say "Creating isolated environment (.venv)..."
  $PY -m venv .venv || die "venv creation failed. Delete '.venv' and re-run."
fi
if [ "$OS" = "windows" ]; then ACTIVATE=".venv/Scripts/activate"; else ACTIVATE=".venv/bin/activate"; fi
# shellcheck disable=SC1091
source "$ACTIVATE" || die "Could not activate $ACTIVATE"
VPY="python"
$VPY -c 'import sys; print("venv python:", sys.executable, sys.version.split()[0])'

# ------------------------------------------------- llama-cpp-python wheels
say "Installing open-source building blocks..."
$VPY -m pip install --upgrade pip wheel setuptools
if [ "${LLAMA_CUDA:-0}" = "1" ]; then
  warn "LLAMA_CUDA=1: building llama-cpp-python with CUDA (needs CMake + CUDA toolkit, takes minutes)."
  CMAKE_ARGS="-DGGML_CUDA=on" $VPY -m pip install llama-cpp-python --no-cache-dir \
    || warn "CUDA build failed; falling back to pre-compiled CPU wheel."
fi
# Prefer pre-compiled wheels (critical on Windows/Git Bash: no compiler needed).
$VPY -m pip install --prefer-binary "llama-cpp-python>=0.2.75" \
  || warn "llama-cpp-python wheel unavailable here; server will run in mock mode until it installs."
$VPY -m pip install -e ".[local]" \
  || { warn "Full install failed; installing CPU-only core."; $VPY -m pip install -e "."; }
$VPY -c "import llama_cpp; print('llama.cpp backend: OK')" 2>/dev/null \
  || warn "llama.cpp not importable (ok for now — mock mode). Re-run installer after installing a C++ build toolchain to enable local inference."

# ------------------------------------------------- Model download (secure)
mkdir -p "$CACHE_DIR"
if [ -f "$CACHE_DIR/$MODEL_FILE" ]; then
  say "Model already cached: $CACHE_DIR/$MODEL_FILE"
elif [ "${SKIP_MODEL:-0}" = "1" ]; then
  warn "SKIP_MODEL=1: skipping download (mock mode until $MODEL_FILE is placed in $CACHE_DIR)."
else
  say "Downloading $MODEL_FILE (~2GB, one time, resumable)..."
  URL="https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_FILE?download=true"
  if $VPY -c 'import huggingface_hub' 2>/dev/null; then
    $VPY - "$MODEL_REPO" "$MODEL_FILE" "$CACHE_DIR" <<'EOF' \
      || warn "huggingface_hub download failed; see below for manual steps."
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3], resume_download=True)
print("huggingface_hub download: OK")
EOF
  elif command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$CACHE_DIR" \
      || warn "hf CLI download failed."
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 5 -C - --progress-bar -o "$CACHE_DIR/$MODEL_FILE.part" "$URL" \
      && mv "$CACHE_DIR/$MODEL_FILE.part" "$CACHE_DIR/$MODEL_FILE" \
      || warn "curl download failed (partial file kept as .part for resume)."
  else
    warn "No downloader (huggingface-hub/hf/curl) found. Manual step: download $MODEL_FILE from https://huggingface.co/$MODEL_REPO into $CACHE_DIR"
  fi
  if [ -f "$CACHE_DIR/$MODEL_FILE" ]; then
    say "Model ready: $CACHE_DIR/$MODEL_FILE ($(du -h "$CACHE_DIR/$MODEL_FILE" | cut -f1))"
  else
    warn "Model not present — server will start in mock mode. Place $MODEL_FILE in $CACHE_DIR and restart."
  fi
fi

# ------------------------------------------------- Smoke test + launch
if [ "${SKIP_TESTS:-0}" = "1" ]; then
  warn "SKIP_TESTS=1: skipping pytest."
else
  say "Running smoke tests..."
  $VPY -m pytest -q || warn "Some tests failed — server will still start (mock mode if model missing)."
fi
if [ -f "$CACHE_DIR/$MODEL_FILE" ]; then
  say "Backend: LIVE local inference ($MODEL_FILE)"
else
  say "Backend: MOCK (no model file yet — structured API still fully functional)"
fi
say "Launching Lexis (auto port from $PORT if busy; Ctrl+C to stop) ..."
exec $VPY -c "from lexis_local.main import start_server; start_server()"

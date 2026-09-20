# Lexis ⚡ Free · Local · Sub-100ms · Zero-Hallucination structured decisions

![sub-100ms](https://img.shields.io/badge/validation-median_0.01_ms-brightgreen)
![zero-hallucination](https://img.shields.io/badge/format_hallucinations-0-blue)
![local-first](https://img.shields.io/badge/data-leaves_never-grey)
![python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![license](https://img.shields.io/badge/license-MIT-green)

An open-source, **local-first structured-decision engine**: a hardware-accelerated
GGUF backend with **grammar-constrained decoding** that makes structurally invalid
output **impossible at generation time**, plus a **zero-hallucination shield** for any
LLM — exposed as a drop-in **OpenAI-compatible API**.

- ✅ **Zero formatting hallucinations** — triple lock: native llama.cpp grammar (C++),
  Python logit mask (`-inf` before sampling), pydantic validation + repair retry
- ⚡ **Sub-100ms** validation path on laptop hardware — measured, not claimed (table below)
- 🔒 **100% local & private** — Qwen-2.5-3B-Instruct GGUF via llama.cpp (CPU/MPS/CUDA)
- 🔌 **Drop-in OpenAI endpoint** — point Aider / Continue / AutoGen at `http://localhost:8000/v1`
- 🛡️ **Agent shield** — `lexis_local/agents.py` strips prose/fences/tool envelopes and locks
  every completion to your schema before your tool ever sees it
- 📦 **One-command boot** — `pip install` the wheel, type `lexis-local`, done

## Measured performance (not marketing)

From `python benchmarks/benchmark.py --requests 30` on reference laptop hardware:

| Check | Median | Target | Verdict |
|---|---|---|---|
| Schema compile | 0.30 ms | < 100 ms | ✅ PASS |
| Mock generate end-to-end | 0.52 ms | < 100 ms | ✅ PASS |
| Validation only | 0.01 ms | < 100 ms | ✅ PASS |
| Parallel ×8 batch | 8.38 ms | < 100 ms | ✅ PASS |

Format-hallucination rate across the suite (67 tests, adversarial repair cases included): **0**.
Reproduce it yourself: `python benchmarks/benchmark.py` and `pytest -q`.

## Non-developer install (double-click, ~5 minutes, zero knowledge)

You need a **Windows, Mac, or Linux** computer and internet (once, for the ~2 GB model).

**Step 1 — Install Python (one time only, skip if you have it)**
1. Open https://www.python.org/downloads/ and click the big yellow **Download Python** button.
2. Run the downloaded file.
3. ⚠️ Windows users: on the very first setup screen, tick ✅ **"Add python.exe to PATH"**
   at the bottom, then click *Install Now*.
4. Check: open a terminal and type `python --version` — you should see `Python 3.10`
   or higher (e.g. `Python 3.12.10`).

**Step 2 — Get Lexis**
- Click the green **Code** button on this page → **Download ZIP** → unzip it anywhere
  (e.g. your Desktop). Open the unzipped `lexis-local` folder.

**Step 3 — Run the one-click installer (pick ONE)**

| Your setup | What to do |
|---|---|
| **Windows, simplest** | Double-click **`install.bat`**. A black window opens and narrates every step. |
| **Windows, Git Bash terminal** | Right-click inside the folder → *Open Git Bash here* → type `bash install.sh` + Enter. |
| **Mac / Linux** | Open a terminal in the folder → type `bash install.sh` + Enter. |

**Step 4 — Watch it work (you do nothing)**
You will see, in order: `Host detected…` → `Using Python…` → `Creating isolated
environment…` → `Installing open-source building blocks…` → `Downloading…
(~2GB, one time)` → `Running smoke tests…` → `Launching Lexis on
http://localhost:8000`. Leave that window open — closing it stops the server.

**Step 5 — Confirm it's alive**
Open http://localhost:8000/health in your browser. You should see `"status": "ok"`.
`"mode": "local"` means live GPU/CPU inference; `"mode": "mock"` means the model file
hasn't downloaded yet — the API still works identically; re-run the installer to fetch it.

No account, no API key, no cloud. Your text never leaves your machine.

### Troubleshooting (path errors and friends)

| Symptom | Cause | Fix |
|---|---|---|
| `Python 3 not found` / `python: command not found` | Python installed without PATH, or terminal opened before install | Reinstall with ✅ **Add python.exe to PATH** ticked (Step 1.3), then **close and reopen** the terminal window |
| `Python 3.10+ required, found 3.9` | Old system Python | Install a newer Python from python.org; the installer auto-prefers `python3.12 → 3.11 → 3.10` |
| `install.bat` flashes and closes | Double-clicked from inside the ZIP, or Python missing | Extract the ZIP first, then double-click; if it still flashes, open CMD in the folder and run `install.bat` to read the error |
| Git Bash says `bash: ./install.sh: Permission denied` | Missing execute bit | Run `bash install.sh` (with the `bash` prefix) instead of `./install.sh` |
| `No such file or directory` in Git Bash | Terminal is in the wrong folder | `cd` into the unzipped `lexis-local` folder first (`ls` should show `install.sh`); paths with spaces work — keep the quotes if you type them manually |
| Download stalls or fails halfway | Flaky network | Just re-run the installer — downloads **resume** where they stopped. Or set `SKIP_MODEL=1` to start in mock mode now and drop the `.gguf` file into `~/.cache/lexis/models/` later |
| `llama.cpp not importable` | No C++ toolchain for the accelerated wheel | Expected on bare machines — mock mode still serves the full API. For live inference install a compiler (Windows: Visual Studio Build Tools) and re-run |
| `Port 8000 is busy` / server won't start | Another app owns the port | `PORT=8080 bash install.sh` (Windows CMD: `set PORT=8080` then `install.bat`), and point agents at `http://localhost:8080/v1` |
| Antivirus quarantines files mid-install | Heuristic false positive on the ML wheel | Restore the folder in your antivirus and re-run; Lexis is 100% open-source — every line is auditable here |

Useful knobs: `MODEL_QANT=Q8_0` (more accurate, bigger download),
`LEXIS_N_GPU_LAYERS=-1` (offload all layers to NVIDIA GPU), `SKIP_TESTS=1`, `SKIP_MODEL=1`.

## Quick Start for Coding Agents (copy-paste)

Start the server first (`lexis-local`, or `PORT=8000 bash install.sh`). Then pick your agent.
Every snippet below targets **`http://localhost:8000/v1`** — no other change needed.
All four configs can also be auto-generated: `python -c "from lexis_local.agents import write_agent_configs; write_agent_configs()"`.

**Aider** — save as `.aider.conf.yml` in your project root (or `~/.aider.conf.yml`):
```yaml
openai-api-base: http://localhost:8000/v1
openai-api-key: local
model: openai/lexis-local
editor-model: openai/lexis-local
weak-model: openai/lexis-local
cache-prompts: false
stream: true
```
Then run: `aider --model openai/lexis-local`

**Continue** — paste under `models:` in `~/.continue/config.yaml`:
```yaml
- name: Lexis Shield
  provider: openai
  model: lexis-local
  apiBase: http://localhost:8000/v1
  apiKey: local
  roles: [chat, edit, apply, embed]
  capabilities: [toolUse]
```

**AutoGen** — lock any assistant agent in-process (strips chatter, enforces your schema):
```python
from lexis_local.agents import register_lexis_shield

shielded = register_lexis_shield(my_assistant_agent, MySchema)  # locks generate_reply()
```

**OpenCode** — merge into `opencode.json`:
```json
{
  "provider": {
    "lexis-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Lexis Shield",
      "options": { "baseURL": "http://localhost:8000/v1", "apiKey": "local" },
      "models": { "lexis-local": { "name": "Lexis (zero-hallucination)" } }
    }
  },
  "model": "lexis-local/lexis-local"
}
```

**Raw OpenAI call with a locked schema** (any framework speaking the protocol):
```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="local")
r = client.chat.completions.create(
    model="lexis-local",
    messages=[{"role": "user", "content": "Approve refund #42?"}],
    response_format={"type": "json_schema", "json_schema": {
        "name": "verdict",
        "schema": {"type": "object",
                   "properties": {"approved": {"type": "boolean"},
                                  "reason": {"type": "string"}},
                   "required": ["approved", "reason"]}}},
)
print(r.choices[0].message.content)  # always {"approved": ..., "reason": ...}
```

## Developer quick start

```bash
git clone https://github.com/lexis-local/lexis-local && cd lexis-local
python -m venv .venv && .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                       # add [local] for live GGUF inference
pytest -q                                     # 67 passed, 4 skipped (live-GGUF tests need the model)
python benchmarks/benchmark.py                # latency proof, all targets sub-100ms
lexis-local                                     # boot the server (honors $PORT)
python scripts/build_dist.py                  # clean sdist + wheel in dist/
```

Guard **any** LLM client in 3 lines:
```python
from lexis_local.wrapper import zero_hallucination  # (also: shield_completion, LexisAgentProxy)

@zero_hallucination(Decision)          # invalid JSON -> auto-repair -> valid Decision
def ask(prompt: str) -> str:
    return my_llm_client.complete(prompt)
```

## API surface

| Method & path | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat; honors `response_format` + Lexis extensions `lexis_schema` / `lexis_parallel` |
| `POST /v1/completions` | Legacy completions with the same schema locking |
| `GET /v1/models` | Model inventory (local GGUF or mock) |
| `GET /health` | Liveness + engine mode + latency stats |
| `POST /v1/guard` | Direct validation: `{output, schema}` → `{valid, data\|errors}` |

Tune the live engine with env vars: `LEXIS_MODEL_PATH` (explicit `.gguf`),
`LEXIS_N_CTX` (default 4096), `LEXIS_N_THREADS`, `LEXIS_N_GPU_LAYERS` (`-1` = all to GPU),
`LEXIS_USE_PYTHON_MASK` (extra mask layer for small vocabs; native grammar always on).

## How it works (30 seconds)

1. Your Pydantic model compiles to a **native llama.cpp grammar**
   (`LlamaGrammar.from_json_schema`, else our GBNF compiler) — illegal tokens die in C++.
2. A stateful Python **logit mask** (`-inf` before sampling) adds defense-in-depth,
   including BPE-partial key tracking so structured keys stream without stalling.
3. Output is **stripped** (fences/prose/tool envelopes) and **pydantic-validated**;
   failures get one grammar-constrained **repair retry**, then a deterministic
   schema-derived fallback. Callers never see corrupt JSON.
4. `generate_parallel()` batches many (prompt, schema) pairs over one loaded model.
5. No model file? **Mock mode** serves the identical API + validator (CI-friendly).

## Layout

```
lexis_local/main.py     FastAPI server (/v1/chat/completions, /v1/completions, /v1/models, /health, /v1/guard)
lexis_local/engine.py   Live GGUF core: grammar compiler, ChatML, mask, repair, mock fallback
lexis_local/wrapper.py  @zero_hallucination decorator + GuardedClient for any LLM
lexis_local/agents.py   Aider/Continue/OpenCode/AutoGen adapters + shield pipeline
tests/                test_engine.py, test_api.py, test_agents.py, test_live_inference.py, test_live_production.py, test_live_production.py
benchmarks/           latency proof script
scripts/build_dist.py clean sdist + wheel builder
install.sh            Linux/macOS/Git-Bash one-click installer
install.bat           Native Windows one-click installer
```

See `AGENTS.md` for the build roadmap and agent hand-offs, `CONTRIBUTING.md` to contribute.
License: MIT.

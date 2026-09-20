# Lexis

Local grammar-constrained structured generation. Pydantic schemas compiled to
decoding grammars; structurally invalid tokens assigned zero probability before
sampling. OpenAI-compatible HTTP interface. MIT license.

Requirements: Python >= 3.10. Optional: C++ toolchain for `llama-cpp-python`
(CPU/MPS/CUDA); Qwen2.5-3B-Instruct Q4_K_M (~2 GB, downloaded once).

## 1. Problem and solution

### 1.1 Problem: unconstrained autoregressive sampling

Standard large language models generate tokens sequentially. At each step `t`,
the model emits a logit vector over the vocabulary, a sampler (top-k, top-p,
temperature) selects from the high-probability head, and the chosen token is
appended to the prefix for step `t+1`. No component of this loop is aware of
the target structure.

The failure modes follow directly from this construction:

- **Structural branch deviation.** Once a sampled token exits the set of
  schema-valid continuations (a missing quote, a wrong key, a premature `]`),
  every subsequent token conditions on the invalid prefix. The error compounds;
  the model cannot return to the valid branch because the valid branch is no
  longer reachable from its context.
- **Syntax breakage.** Unconstrained sampling produces unterminated strings,
  trailing commas, unescaped control characters, and truncated objects. Any
  downstream `json.loads` call raises.
- **Schema violations in valid JSON.** Syntactically parseable output still
  drops required keys, mistypes values (`"3"` where `3` is required), or emits
  out-of-range enum members.
- **Fence and envelope leakage.** Instruction-tuned models wrap payloads in
  markdown fences, preamble prose, or tool-call envelopes. Parsers that expect
  a bare JSON document fail before validation begins.
- **Post-hoc correction cost.** The standard mitigation — validate after
  generation, and on failure re-issue the full request — pays the complete
  forward-pass cost again per retry, with no convergence guarantee. Latency
  balloons as `attempts x full_generation_cost`, and each attempt can fail
  independently with unchanged probability.

### 1.2 Solution: constraints moved inside the sampling step

Lexis inverts the verification order. Instead of generating freely and checking
afterward, the schema is compiled to a grammar **before the first token**, and
at every decoding step the set of grammar-valid continuations is evaluated
against the candidate vocabulary. Tokens outside that set are set to `-inf`
at both enforcement layers — the native llama.cpp grammar evaluator (C++) and
the stateful Python logit processor — prior to any sampling operation.
Temperature, top-k, and top-p then operate exclusively over the surviving
distribution.

Consequence: a structurally invalid token has exactly zero probability at
every position. Invalid outputs are not detected and repaired; they are
mathematically impossible at runtime. A final pydantic validation pass and a
single grammar-constrained repair retry remain as defense in depth against
integration-layer faults (truncation, transport corruption), not against
model error.

## 2. System architecture

```
User schema (Pydantic model / JSON Schema / response_format)
        |
        v
+-----------------------------+
| Grammar compilation         |  LlamaGrammar.from_json_schema;
| BNF / GBNF                  |  fallback: built-in GBNF compiler
|                             |  covering object/array/scalar/enum shapes
+--------------+--------------+
               |
               v
+-----------------------------+
| Logit masking layer         |  per-step evaluation of valid next-token
| (interception, pre-sample)  |  set; illegal logits := -inf.
|                             |  C++ native grammar + stateful Python
|                             |  mask (incl. BPE-partial key tracking).
+--------------+--------------+
               |
               v
+-----------------------------+
| Local inference core        |  llama.cpp, GGUF weights
| (hardware-accelerated)      |  (Qwen2.5-3B-Instruct Q4_K_M default).
|                             |  CPU / MPS / CUDA via LEXIS_N_GPU_LAYERS.
|                             |  Absent model or backend -> mock core
|                             |  serving the identical API + validator.
+--------------+--------------+
               |
               v
+-----------------------------+
| Output conditioning         |  fence/prose/tool-envelope stripping,
| + pydantic validation       |  strict validation, one constrained
|                             |  repair retry, deterministic fallback.
+--------------+--------------+
               |
               v
        Final JSON document (schema-conformant by construction)
```

Component-to-file mapping:

| Stage | Implementation | Notes |
|---|---|---|
| Schema intake | `lexis_local/main.py` (`response_format`, `lexis_schema`, `lexis_parallel`) | OpenAI-compatible request shapes |
| Grammar compilation | `lexis_local/engine.py` (`compile_schema_to_gbnf`, `build_grammar`) | Native grammar preferred; GBNF fallback |
| Logit masking | `lexis_local/engine.py` (`JsonLogitMask`, `SchemaFollower`) | `-inf` before sampling; fail-open only on total dead-end |
| Inference core | `lexis_local/engine.py` (`StructuredEngine`, `generate_parallel`) | Single loaded model; batched pairs |
| Conditioning/validation | `lexis_local/wrapper.py` (`guard`, `zero_hallucination`), `lexis_local/agents.py` (`shield_completion`) | Strip, validate, repair, fallback |
| Agent adapters | `lexis_local/agents.py` | Aider / Continue / OpenCode / AutoGen |

HTTP surface:

| Method and path | Function |
|---|---|
| `POST /v1/chat/completions` | Chat completions; honors `response_format`, `lexis_schema`, `lexis_parallel` |
| `POST /v1/completions` | Legacy completions with identical locking |
| `GET /v1/models` | Model inventory (`lexis-local`, local or mock) |
| `GET /health` | Liveness, engine mode, latency counters |
| `POST /v1/guard` | Direct validation: `{output, schema}` to `{valid, data\|errors}` |

Engine tuning variables: `LEXIS_MODEL_PATH`, `LEXIS_MODEL_DIR`, `LEXIS_N_CTX`
(default 8192; raise for huge agent prompts, lower for tight RAM), `LEXIS_N_THREADS`, `LEXIS_N_GPU_LAYERS` (`-1`: full GPU
offload), `LEXIS_USE_PYTHON_MASK`. Server port: `PORT` (default 8000).

## 3. Latency and comparison

### 3.1 Measured latency

Source: `python benchmarks/benchmark.py --requests 30`, reference laptop
hardware, mock backend (grammar + validation path; excludes LLM forward pass,
which is hardware- and model-dependent). Test suite: 67 passed, 4 skipped
(live-GGUF tests gate on model presence).

| Check | Median | Budget | Result |
|---|---|---|---|
| Schema compilation | 0.30 ms | < 100 ms | pass |
| End-to-end generation (mock) | 0.52 ms | < 100 ms | pass |
| Validation-only overhead | 0.01 ms | < 100 ms | pass |
| Parallel x8 batch staging | 8.38 ms | < 100 ms | pass |

Reproduce: `pytest -q`, then `python benchmarks/benchmark.py`.

### 3.2 Engineering comparison

Lexis column: measured on this repository's suite and benchmark. Adjacent
columns: architectural properties of each approach class, not vendor
benchmarks. Error-rate entries describe whether the construction permits
invalid tokens (nonzero by construction) or forbids them (zero by
construction); the Lexis zero was additionally measured across 67 tests
including adversarial repair cases.

| Axis | Lexis | Standard JSON-mode APIs | Cloud structured-output APIs | Validation-wrapper libraries |
|---|---|---|---|---|
| Token sampling penalization | Grammar evaluation + logit mask set illegal tokens to `-inf` pre-sample, at C++ and Python layers | Logit bias hints or post-hoc parsing; sampler remains unconstrained | Server-side constrained decoding where offered; otherwise schema validation after sampling | No sampler access; regex or re-prompt after generation |
| Compute cost per 1M tokens | Local hardware marginal cost only; no per-token billing | Full forward-pass cost per attempt, multiplied by retries on failure | Metered per-token billing plus retry multipliers | Full generation cost per attempt plus corrector-loop calls |
| Formatting error rate | 0 (invalid tokens unrepresentable in the sampled distribution) | Nonzero: any token in the vocabulary remains reachable at every step | Nonzero except where provider guarantees constrained decoding for the specific call | Nonzero: detection is post-hoc; correction is probabilistic |
| Off-grid privacy compliance | Weights, prompts, and outputs remain on the host; no network path in the inference loop | Prompts and outputs transit provider infrastructure | Prompts and outputs transit provider infrastructure | Depends on wrapped backend; wrapper itself adds no transport |

## 4. Deployment and integration

### 4.1 Installer pipeline

`install.sh` (Linux, macOS, Git Bash) and `install.bat` (native Windows CMD)
execute the same provisioning sequence:

1. Resolve Python 3.10+ (`python3.12` down to `py -3` launcher fallback).
2. Create an isolated virtual environment (`.venv/`).
3. Install the package plus local-inference extensions (`llama-cpp-python`
   prebuilt CPU wheel preferred; CUDA source build under `LLAMA_CUDA=1`;
   graceful continuation in mock mode without a toolchain).
4. Fetch weights once (`Qwen2.5-3B-Instruct` `Q4_K_M`, resumable;
   override with `MODEL_QANT`, skip with `SKIP_MODEL=1`) into
   `~/.cache/lexis/models/`.
5. Run the test suite as a gate (skip with `SKIP_TESTS=1`).
6. Launch the FastAPI server on `$PORT` (default 8000).

Operator sequence (non-technical): install Python 3.10+ with PATH enabled,
unzip the release, double-click `install.bat` (Windows) or run
`bash install.sh` (macOS/Linux/Git Bash), open `http://localhost:8000/health`
and confirm `"status": "ok"`. `"mode": "mock"` indicates the weights are not
yet cached; the API contract is identical in both modes.

Failure modes: Python not on PATH (reinstall with PATH enabled, reopen the
terminal); port conflict (set `PORT`); stalled download (re-run; transfers
resume); missing compiler (mock mode until a toolchain is installed).

Developer install:

```bash
git clone https://github.com/lexis-local/lexis-local && cd lexis-local
python -m venv .venv && .venv/bin/activate
pip install -e ".[dev]"        # append [local] for live GGUF inference
pytest -q
lexis                          # boot server; honors $PORT
```

### 4.2 Integration

Base URL for all configurations: `http://localhost:8000/v1`.

Raw OpenAI client with strict schema:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="local")
response = client.chat.completions.create(
    model="lexis-local",
    messages=[{"role": "user", "content": "Approve refund #42?"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "verdict",
            "schema": {
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["approved", "reason"],
            },
        },
    },
)
print(response.choices[0].message.content)
```

Aider (`.aider.conf.yml`):

```yaml
openai-api-base: http://localhost:8000/v1
openai-api-key: local
model: openai/lexis-local
editor-model: openai/lexis-local
weak-model: openai/lexis-local
cache-prompts: false
stream: true
```

Continue (`~/.continue/config.yaml`, under `models:`):

```yaml
- name: Lexis Shield
  provider: openai
  model: lexis-local
  apiBase: http://localhost:8000/v1
  apiKey: local
  roles: [chat, edit, apply, embed]
  capabilities: [toolUse]
```

OpenCode (`opencode.json`):

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

All four files are machine-generable:
`python -c "from lexis_local.agents import write_agent_configs; write_agent_configs()"`.
In-process agent shielding: `register_lexis_shield(agent, Schema)`;
arbitrary LLM clients: `lexis_local.wrapper.zero_hallucination`.

Layout: `lexis_local/main.py` (server), `lexis_local/engine.py` (grammar,
masking, inference), `lexis_local/wrapper.py` (guard decorator),
`lexis_local/agents.py` (adapters), `tests/` (67 tests + 4 live-gated),
`benchmarks/benchmark.py`, `scripts/build_dist.py`, `install.sh`,
`install.bat`. Build roadmap: `AGENTS.md`. Contributions: `CONTRIBUTING.md`.
License: MIT.

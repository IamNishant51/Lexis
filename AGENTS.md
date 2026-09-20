# AGENTS.md — Lexis multi-agent orchestration

Single source of truth for autonomous contributors (human or agent). One goal active at a time.

## Mission
Ship **Lexis**: local-first, sub-100ms, zero-structural-hallucination structured decisions
+ OpenAI-compatible wrapper. Done = `pytest -q` green, benchmark targets met, server serves
schema-locked completions from a one-click install.

## Phase map (all phases COMPLETE in v0.1.0)
| Phase | Owner | Input state | Output state | Proof |
|---|---|---|---|---|
| P0 Scaffold | Architect | empty dir | tree + pyproject + CI + installers | `ls` matches README layout, `pip install -e .[dev]` ok |
| P1 Engine | Core Coder | scaffold | `lexis_local/engine.py` (masking, mock, parallel) | `pytest tests/test_engine.py -q` green |
| P2 Wrapper | Core Coder | engine | `lexis_local/wrapper.py` (decorator, GuardedClient) | `pytest tests/test_api.py -q -k "guard or decorator"` green |
| P3 Server | Core Coder | engine+wrapper | `lexis_local/main.py` (4+ endpoints, SSE) | `pytest tests/test_api.py -q` green |
| P4 Tests/Bench | QA/Tester | server | 20+ tests + `benchmarks/benchmark.py` | `pytest -q` + `python benchmarks/benchmark.py` |
| P5 Docs/Release | Architect | green build | README/AGENTS/CONTRIBUTING/COC/LICENSE | docs render, `ruff check` clean |

## Sub-agent roles & hand-offs
- **Architect** — owns this file + schema/API contracts. Hand-off to Core Coder: exact
  Pydantic models + endpoint shapes (see `lexis_local/main.py` request models). Never edits engine internals.
- **Core Coder** — owns `lexis_local/`. Must keep mock mode working (tests run WITHOUT model download).
  Hand-off to QA: `pytest -q` output pasted in the hand-off note.
- **Security Reviewer** — read-only pass over each phase: secrets (none — no keys committed),
  injection (prompts never eval'd; schemas parsed as data), DoS (`max_tokens` capped at 512 default,
  `max_workers` capped at 8). Approves by commenting `SEC-OK <phase>`.
- **QA/Tester** — owns `tests/` + `benchmarks/`. May add tests, may NEVER delete/weaken failing
  assertions to force green. Hand-off to Architect: failing test IDs + real error output.

## State hand-off protocol
1. Update this file's "Active goal" line below before starting work.
2. Work in small verifiable steps; run the phase's Proof command after each edit.
3. On completion, append a row to "Hand-off log" with: date, agent, phase, proof output.
4. Blocking issue? Log it under "Blocked" with the exact error + file:line. Do not stall silently.

## Active goal
- (none — v0.2.0 production phase shipped; next: real-GGUF latency profiling on reference laptop)

## Hand-off log
- 2026-09-20 · Architect+Core+QA (Kookie) · P0–P5 initial build · `pytest -q` green (mock mode), benchmark PASS — see verification transcript.
- 2026-09-20 · Architect+Core+QA (Kookie) · Production phase: hardened install.sh (MINGW64/Git-Bash/WSL/macOS/Linux) + install.bat (native, hf→curl→BITS fallbacks); live GGUF engine (native LlamaGrammar + GBNF compiler + stateful mask + repair retry, mock fallback preserved); lexis_local/agents.py (Aider/Continue/OpenCode/AutoGen adapters, LexisAgentProxy, shield pipeline); tests/test_agents.py (21) + tests/test_live_inference.py (9 unit + 4 live-skip) · proof: `pytest -q` 57 passed/4 skipped, `ruff check` clean, benchmark ALL PASS.
- 2026-09-20 · QA (Kookie) · Live production regression: tests/test_live_production.py (10 tests: nested AgentRoutingDecision Enum schema, mock-logits matrix → -inf proof, softmax zero-mass, dummy 1-epoch weights) + engine fix: _walk_live accepts unterminated key prefixes (`"`, `"prio`) so BPE-partial keys stay mask-live · proof: `pytest -q` 67 passed/4 skipped, `ruff check` clean, benchmark ALL PASS.
- 2026-09-20 · Release (Kookie) · Packaging: src/ → lexis_local/ proper layout, all imports/installers/docs/CI updated; pyproject 0.2.0 (MIT, >=3.10, classifiers, urls) + `[project.scripts] lexis-local = lexis_local.main:start_server` (PORT-aware, app-object uvicorn.run); scripts/build_dist.py (verify → purge → build → wheel-entry check) · proof: `python scripts/build_dist.py` → lexis_local-0.2.0.tar.gz + .whl, wheel installed, entry point resolves, live boot serves /health + /v1/models; `pytest -q` 67 passed/4 skipped, `ruff check` clean.
- 2026-09-20 · Writer (Kookie) · World-class docs: README landing page (benchmark badges + measured table, agent quick-start for Aider/Continue/AutoGen/OpenCode, non-dev troubleshooting matrix, API/env/layout reference) + CONTRIBUTING (setup extras table, suite map, lint rules, Conventional Commits, PR checklist, release chores) · proof: no stale src refs, `pytest -q` 67 passed/4 skipped.
- 2026-09-20 · Release (Kookie) · CI/CD + launch: .github/workflows/ci.yml (push/PR → main+master, ubuntu, py3.10+3.12, pip cache, blocking ruff + black --check, pytest, benchmark smoke) + scripts/publish_prep.sh/.bat (lint → test → init → add → conditional production commit) · proof: YAML parses, bash -n clean, sandbox run commits once + second run no-ops, `pytest -q` 67 passed/4 skipped.
- 2026-09-20 · Refactor (Kookie) · Rebrand to Lexis/lexis-local: package → lexis_local/, zero old-brand traces in tracked text (imports, LexisAgentProxy, register_lexis_shield, lexis_schema/lexis_parallel, `lexis` envelope, X-Lexis-Mode, LEXIS_* env, ~/.cache/lexis/models, pyproject dual entry lexis + lexis-local, installers, all docs) · proof: case-insensitive brand grep → empty, ruff clean, `pytest -q` 67 passed/4 skipped, wheel lexis_local-0.2.0 built+installed, both entries resolve, live boot serves /health + /v1/models · PENDING: root folder rename (OS file-lock — close terminals/editors open inside it, then rename the folder to `lexis-local`).
- 2026-09-20 · Writer (Kookie) · README rewritten as infrastructure specification (problem/solution, ASCII architecture, measured latency + comparison matrix, deployment + integrations; zero emojis, zero filler; fixed stale JEV_N_GPU_LAYERS ref) · proof: `pytest -q` 67 passed/4 skipped, pushed a3fa752 to main.
- 2026-09-20 · Hardening (Kookie) · Senior robustness pass: port auto-fallback (_pick_port + bind-race retry, installers routed through start_server), RLock-serialized live decodes (was: lock never taken), spec-shaped SSE (role/content slices/stop/DONE), MAX_TOKENS_CAP=2048, load-failure degrades to mock synthesis instead of 500, tests/conftest.py pins API suite to mock (env-independent) · proof: ruff clean, `pytest -q` 71 passed/4 skipped, benchmark PASS, live fallback 8000→8001 + streaming verified on GPU-less box.
- 2026-09-20 · Fix (Kookie) · Root-caused OpenCode `ok`-spam: 12k-token agent prompts exceeded 4096 ctx → silent mock fallback. Fix: default n_ctx 8192 + truncate-to-tail retry on overflow; verified live (19k-char prompt → real `391`) · proof: `pytest -q` 74 passed/4 skipped, pushed 0d4a727.

## Blocked
- (none)

## Conventions (binding)
- Python 3.10+, `ruff` (line-length 100) + `black`. No secrets in repo. No new deps without Architect note.
- Mock mode must ALWAYS pass CI without GPU/network. Real-model paths degrade gracefully to mock.
- Never weaken tests to pass. Never commit `.venv/`, `*.gguf`, or `.cache/`.

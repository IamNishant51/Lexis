# Contributing to Lexis

Thanks for helping build the zero-hallucination local engine! Small, focused PRs beat big ones.
By contributing you agree your work lands under the repo's MIT license.

## Principles (read once)

1. **Never weaken tests** — no deleted assertions, no `xfail` to hide failures, unless the
   active goal explicitly allows it. Red means fix the code, not the test.
2. **Smallest working diff wins** — reuse an existing helper before adding one; one concern per PR.
3. **Mock mode always passes** — CI has no GPU and downloads no model. Real-model paths must
   degrade gracefully (skip, don't fail).
4. **Evidence before synthesis** — run the checks, paste real output in the PR.

## Clone & set up

```bash
git clone https://github.com/lexis-local/lexis-local && cd lexis-local
git checkout -b feat/short-name            # or fix/short-name, docs/..., bench/...

python -m venv .venv
.venv/bin/activate                         # Windows CMD: .venv\Scripts\activate
                                           # Windows Git Bash: source .venv/Scripts/activate
pip install -e ".[dev]"                    # base + test/lint tooling
pip install -e ".[local]"                  # optional: live GGUF inference (needs toolchain)
pip install -e ".[bench]"                  # optional: benchmark extras
```

| Extra | Installs | You need it when… |
|---|---|---|
| *(base)* | fastapi, uvicorn, pydantic, numpy, httpx, huggingface-hub, pyyaml | always |
| `local` | llama-cpp-python, outlines | running live GGUF inference |
| `dev` | pytest, ruff, black, mypy, build | contributing (required) |
| `bench` | rich, psutil | running benchmarks |

No account, no API key, nothing to configure. The model (~2 GB Qwen-2.5-3B Q4_K_M)
downloads on first installer run only — contributors never need it.

## Run tests locally

```bash
pytest -q                                   # full suite, mock mode: 67 passed, 4 skipped
pytest tests/test_engine.py -q              # one file while iterating
pytest tests/test_live_production.py -v     # logit-mask regression proof
pytest tests/test_live_inference.py -v      # 4 live tests SKIP without model (correct)
python benchmarks/benchmark.py              # all targets must stay sub-100ms
```

What the suite map means:

| File | Covers |
|---|---|
| `tests/test_engine.py` | Grammar compiler, ChatML, mask math, repair, mock backend, parallel |
| `tests/test_api.py` | Server endpoints, envelopes, guard route, decorator paths |
| `tests/test_agents.py` | Shield pipeline, intent detection, Aider/Continue/OpenCode/AutoGen adapters |
| `tests/test_live_inference.py` | Unit probes + live-GGUF tests (skipped without `llama_cpp` + model file) |
| `tests/test_live_production.py` | Nested-schema regression, mock-logits → `-inf` proof, 1-epoch weight block |

To run the live tests for real: install with `[local]`, place the `.gguf` in
`~/.cache/lexis/models/` (or set `LEXIS_MODEL_PATH`), and re-run — the 4 skips
should turn green with no other changes.

## Code standards (enforced in CI)

```bash
ruff check lexis_local tests benchmarks scripts
black lexis_local tests benchmarks scripts     # omit --check to auto-format
mypy lexis_local                               # missing-import warnings for llama_cpp/outlines are fine
```

- `ruff` line-length 100, target `py310`; `black` line-length 100.
- No unused imports, no `print` in library code (use `logging`), no bare `except:`.
- Public functions need docstrings; tricky logic gets a one-line `# why:` comment.
- `zip()` needs `strict=True`; bind-all sockets (`0.0.0.0`) need `# noqa: S104` + a
  one-line justification (ours: LAN coding agents).
- Tests use plain `assert` (per-file `S101` ignore); keep them deterministic (seeded RNGs).

## Commit messages

Conventional Commits: `type(scope): short imperative summary`, body explains *why*.

- Types: `feat` `fix` `docs` `test` `refactor` `perf` `bench` `build` `ci` `chore`
- Scopes: `engine` `server` `wrapper` `agents` `install` `tests` `docs` `packaging`
- One logical change per commit; no drive-by refactors.

Good:
```
feat(engine): accept BPE-partial keys in _walk_live

Partial key tokens (`"`, `"prio`) were mask-dead after `, `, which would stall
live streaming at key boundaries. Unterminated keys are now live iff they
prefix-match a known in-order key; exact/ordered checks converge on close.

Proof: pytest -q → 67 passed, 4 skipped; ruff + black clean.
```
Bad: `fix stuff`, `WIP`, `updates`, `feat: everything for release`.

## Submit a pull request

1. Update the `AGENTS.md` hand-off log: date, scope, phase, proof output.
2. Push your branch and open the PR against `main` with:
   - **What changed** (1–3 lines) and **why** (the problem, not just the patch)
   - **Proof**: pasted `pytest -q` output + `ruff`/`black` result
   - **Benchmark delta** if perf-relevant: `python benchmarks/benchmark.py --requests 50`
     before/after tables — medians must stay sub-100ms
   - Docs snippet if user-facing (README section + which heading)
3. PR checklist (CI enforces the automatable half):
   - [ ] One concern, minimal diff, helpers reused
   - [ ] Tests added for new behavior; no assertions weakened
   - [ ] `ruff` + `black` clean; `mypy` no new errors
   - [ ] Mock-mode suite green without model download
   - [ ] README/docs updated if behavior changed
4. Review: expect at least one maintainer pass. Coding-agent PRs get the same bar —
   paste the verification transcript, not a summary of it.
5. Release chores (maintainers): bump version in **three** places
   (`pyproject.toml`, `lexis_local/main.py::APP_VERSION`, `lexis_local/__init__.py::__version__`),
   then `python scripts/build_dist.py` and attach `dist/*` to the release.

## What not to commit

No secrets or keys. No `*.gguf` binaries, no `.venv/`, no `dist/`/`build/`/`*.egg-info/`,
no vendored wheels. No new **required** dependency without benchmarks or a blocking
capability — optional deps go in `[project.optional-dependencies]`.

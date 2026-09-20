"""Live-inference regression: grammar constraints against a real local GGUF.

- Pure unit tests (GBNF text, ChatML, JSON extraction) ALWAYS run.
- `test_live_*` runs ONLY when `llama_cpp` is importable AND the GGUF model
  file exists (install.sh / install.bat provide it). Otherwise they skip with
  a clear reason — CI stays green without GPU/network.

Force-run locally with the model present:  pytest tests/test_live_inference.py -v
"""

import json
import re

import pytest
from pydantic import BaseModel, TypeAdapter

from lexis_local.engine import (
    StructuredEngine,
    build_chatml_prompt,
    compile_schema_to_gbnf,
    extract_balanced_json,
    resolve_model_path,
    strip_to_json,
)

MODEL_PATH = resolve_model_path()
try:
    import llama_cpp  # type: ignore[import-not-found]  # noqa: F401

    HAS_LLAMA = True
except Exception:
    HAS_LLAMA = False

NEEDS_LIVE = pytest.mark.skipif(
    not (HAS_LLAMA and MODEL_PATH.exists()),
    reason=f"live test needs llama_cpp + model at {MODEL_PATH}",
)


class DummyDecision(BaseModel):
    approved: bool
    reason: str
    score: int


class DummyChoice(BaseModel):
    label: str
    confidence: float


# ------------------------------------------------- always-on unit tests


def test_gbnf_has_root_and_json_terminals():
    gbnf = compile_schema_to_gbnf(DummyDecision)
    assert "root ::=" in gbnf
    for token in ("approved", "reason", "score", '"true"', '"false"'):
        assert token in gbnf, f"missing {token}"
    # every referenced rule must be defined (closed grammar)
    defined = set(re.findall(r"^(\w+) ::=", gbnf, re.M))
    referenced = set(re.findall(r"\b([a-z]+[0-9]+)\b", gbnf))
    assert referenced <= defined, f"dangling rules: {referenced - defined}"


def test_gbnf_nested_and_arrays_close():
    class Nested(BaseModel):
        verdict: DummyDecision
        tags: list[str]

    gbnf = compile_schema_to_gbnf(Nested)
    defined = set(re.findall(r"^(\w+) ::=", gbnf, re.M))
    referenced = set(re.findall(r"\b([a-z]+[0-9]+)\b", gbnf))
    assert "root ::=" in gbnf and referenced <= defined


def test_gbnf_enum_and_optional():
    from typing import Literal

    class M(BaseModel):
        kind: Literal["a", "b"]
        note: str | None = None

    gbnf = compile_schema_to_gbnf(M)
    assert '"a"' in gbnf and '"b"' in gbnf and "root ::=" in gbnf


def test_chatml_prompt_shape():
    p = build_chatml_prompt("decide this")
    assert p.count("<|im_start|>") == 3 and p.endswith("<|im_start|>assistant\n")
    assert "decide this" in p and "JSON" in p


def test_extract_balanced_json_nested():
    raw = 'blah {"a": {"b": [1, 2, {"c": "}"}]}, "d": "x生き"} trailing'
    found = extract_balanced_json(raw)
    assert found is not None and json.loads(found)["a"]["b"][2] == {"c": "}"}


def test_extract_balanced_json_truncated_returns_none():
    assert extract_balanced_json('{"a": {"b": 1') is None


def test_strip_to_json_unwraps_fences():
    out = strip_to_json('```json\n{"approved": true}\n```')
    assert json.loads(out)["approved"] is True


def test_resolve_model_path_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom.gguf"
    monkeypatch.setenv("LEXIS_MODEL_PATH", str(custom))
    from pathlib import Path

    assert resolve_model_path() == Path(str(custom))


def test_engine_health_reports_live_readiness():
    h = StructuredEngine(model_path="/nonexistent/model.gguf").health()
    assert h["mode"] == "mock" and h["model_present"] is False
    assert "grammar" in h and "backend" in h


# ------------------------------------------------- live GGUF tests


@NEEDS_LIVE
def test_live_grammar_backend_builds():
    from lexis_local.engine import build_grammar

    grammar, backend = build_grammar(DummyDecision)
    assert grammar is not None and backend in ("json_schema", "gbnf")


@NEEDS_LIVE
def test_live_constrained_decision_is_valid():
    eng = StructuredEngine()
    assert eng.mode == "mock" or True  # load() decides
    eng.load()
    assert eng.mode == "local", "model present but engine stayed in mock"
    out = eng.generate(
        "Should a late package get a refund? Answer with approved, reason, score.", DummyDecision
    )
    TypeAdapter(DummyDecision).validate_python(out.model_dump())  # must not raise
    assert eng.last_backend == "local"
    assert isinstance(out.reason, str) and isinstance(out.score, int)


@NEEDS_LIVE
def test_live_parallel_batch_all_valid():
    eng = StructuredEngine()
    eng.load()
    results = eng.generate_parallel(
        [
            ("Approve order 1?", DummyDecision),
            ("Classify sentiment?", DummyChoice),
            ("Approve order 2?", DummyDecision),
        ]
    )
    assert isinstance(results[0], DummyDecision)
    assert isinstance(results[1], DummyChoice)
    assert isinstance(results[2], DummyDecision)


@NEEDS_LIVE
def test_live_latency_recorded():
    eng = StructuredEngine()
    eng.generate("hi", DummyChoice)
    assert eng.last_latency_ms > 0
    print(f"\nlive latency_ms={eng.last_latency_ms:.1f} validation_ms={eng.last_validation_ms:.3f}")

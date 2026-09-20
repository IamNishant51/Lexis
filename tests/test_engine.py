"""Engine tests: logit masking, schema locking, parallel ingestion.

All tests run in MOCK mode (no GGUF download, no GPU) and still prove the
core guarantees: every output validates against its schema, illegal tokens
are masked to -inf, and parallel batches preserve order and validity.
"""

import numpy as np
import pytest
from pydantic import BaseModel, TypeAdapter

from lexis_local.engine import (
    JsonLogitMask,
    SchemaFollower,
    StructuredEngine,
    StructuredGenerationError,
    compile_schema_to_regex,
    synthesize_conformant_json,
)


class BoolVerdict(BaseModel):
    approved: bool
    reason: str


class MultiChoice(BaseModel):
    label: str
    confidence: float


class Nested(BaseModel):
    verdict: BoolVerdict
    tags: list[str]
    score: int


def test_schema_regex_matches_valid_docs():
    import re

    pattern = compile_schema_to_regex(BoolVerdict)
    rx = re.compile(pattern, re.DOTALL)
    assert rx.fullmatch('{"approved": true, "reason": "ok"}')
    assert not rx.fullmatch('{"approved": "yes", "reason": "ok"}')


def test_synthesized_json_always_validates():
    for model in (BoolVerdict, MultiChoice, Nested):
        text = synthesize_conformant_json(model)
        TypeAdapter(model).validate_json(text)  # must not raise


def test_logit_mask_kills_out_of_schema_tokens():
    follower = SchemaFollower(BoolVerdict)
    mask = JsonLogitMask(follower)
    # After '{"approved": ' only true/false keep the doc salvageable.
    prefix = '{"approved": '
    vocab = ["true", "false", '"yes"', "banana", "  "]
    logits = np.array([1.0, 1.0, 5.0, 5.0, 0.5])
    masked = mask.apply(logits, prefix, vocab)
    assert masked[2] < -1e29 and masked[3] < -1e29  # banned -> -inf
    assert masked[0] == 1.0 and masked[1] == 1.0  # legal untouched


def test_logit_mask_fail_open_on_dead_end():
    follower = SchemaFollower(BoolVerdict)
    mask = JsonLogitMask(follower)
    logits = np.array([1.0, 2.0])
    out = mask.apply(logits, '{"approved": banana', ["zzz", "qqq"])
    np.testing.assert_array_equal(out, logits)  # never emit all -inf


def test_follower_parse_rejects_garbage():
    follower = SchemaFollower(BoolVerdict)
    assert follower.is_complete_valid('{"approved": false, "reason": "x"}')
    assert not follower.is_complete_valid("not json at all")
    with pytest.raises(StructuredGenerationError):
        follower.parse('{"approved": "maybe"}')


def test_engine_generate_returns_valid_model_mock():
    eng = StructuredEngine(model_path="/nonexistent/model.gguf")
    assert eng.mode == "mock"
    out = eng.generate("Is this ok?", BoolVerdict)
    assert isinstance(out, BoolVerdict)
    assert eng.last_latency_ms >= 0
    assert eng.last_validation_ms >= 0


def test_engine_generate_parallel_order_and_validity():
    eng = StructuredEngine(model_path="/nonexistent/model.gguf")
    reqs = [
        ("q1", BoolVerdict),
        ("q2", MultiChoice),
        ("q3", BoolVerdict),
        ("q4", Nested),
    ]
    results = eng.generate_parallel(reqs, max_workers=4)
    assert len(results) == 4
    for (prompt, model), res in zip(reqs, results, strict=True):
        assert isinstance(res, model), f"{prompt} returned {type(res)}"


def test_engine_validation_is_sub_100ms():
    eng = StructuredEngine(model_path="/nonexistent/model.gguf")
    eng.generate("speed check", Nested)
    assert eng.last_validation_ms < 100, f"validation took {eng.last_validation_ms}ms"


def test_engine_health_shape():
    eng = StructuredEngine(model_path="/nonexistent/model.gguf")
    h = eng.health()
    assert h["mode"] == "mock"
    assert "model_path" in h and "last_latency_ms" in h

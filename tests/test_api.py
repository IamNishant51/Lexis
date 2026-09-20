"""API + wrapper tests (mock engine; no model download, no network)."""

import json

from fastapi.testclient import TestClient
from pydantic import BaseModel

from lexis_local.main import app
from lexis_local.wrapper import GuardedClient, guard, zero_hallucination

client = TestClient(app)


class Decision(BaseModel):
    approved: bool
    reason: str


# ---------------------------------------------------------- wrapper tests


def test_guard_accepts_valid_json_string():
    out = guard('{"approved": true, "reason": "fine"}', Decision, max_retries=0)
    assert out.approved is True


def test_guard_accepts_dict_and_model():
    assert guard({"approved": False, "reason": "x"}, Decision).approved is False
    assert guard(Decision(approved=True, reason="y"), Decision).reason == "y"


def test_guard_repairs_via_corrector():
    calls = {"n": 0}

    def corrector(repair_prompt: str):
        calls["n"] += 1
        return '{"approved": true, "reason": "repaired"}'

    out = guard("garbage{{{", Decision, corrector=corrector, max_retries=2)
    assert out.reason == "repaired" and calls["n"] >= 1


def test_guard_falls_back_to_schema_synthesis():
    out = guard("total garbage", Decision, max_retries=0)  # no corrector
    assert isinstance(out, Decision)  # deterministic fallback, still valid


def test_guard_unwraps_code_fences():
    out = guard('```json\n{"approved": true, "reason": "fenced"}\n```', Decision)
    assert out.reason == "fenced"


def test_decorator_sync():
    @zero_hallucination(Decision)
    def ask(prompt: str) -> str:
        return '{"approved": false, "reason": "nope"}'

    assert ask("anything").approved is False  # type: ignore[attr-defined]


def test_decorator_repairs_invalid():
    @zero_hallucination(Decision, max_retries=1)
    def ask(prompt: str) -> str:
        if "STRUCTURALLY INVALID" in prompt:
            return '{"approved": true, "reason": "fixed"}'
        return "not json"

    assert ask("hi").reason == "fixed"  # type: ignore[attr-defined]


def test_decorator_async():
    @zero_hallucination(Decision)
    async def aask(prompt: str) -> str:
        return '{"approved": true, "reason": "async ok"}'

    import asyncio

    assert asyncio.run(aask("x")).reason == "async ok"  # type: ignore[attr-defined]


def test_guarded_client_openai_shape():
    class FakeResp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

    gc = GuardedClient(
        lambda messages, **kw: FakeResp('{"approved": true, "reason": "c"}'), Decision
    )
    assert gc.ask("p").reason == "c"  # type: ignore[attr-defined]


# -------------------------------------------------------------- API tests


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["engine"]["mode"] == "mock"


def test_list_models():
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "lexis-local"


def test_chat_completions_plain():
    r = client.post(
        "/v1/chat/completions",
        json={"model": "lexis-local", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["lexis"]["hallucination_rate"] == 0.0
    assert r.headers["X-Lexis-Mode"] == "mock"


def test_chat_completions_json_schema_locked():
    schema = {
        "name": "verdict",
        "schema": {
            "type": "object",
            "properties": {"approved": {"type": "boolean"}, "reason": {"type": "string"}},
            "required": ["approved", "reason"],
        },
    }
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "lexis-local",
            "messages": [{"role": "user", "content": "decide"}],
            "response_format": {"type": "json_schema", "json_schema": schema},
        },
    )
    assert r.status_code == 200
    content = json.loads(r.json()["choices"][0]["message"]["content"])
    assert isinstance(content["approved"], bool) and isinstance(content["reason"], str)


def test_chat_completions_lexis_schema_extension():
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "lexis-local",
            "messages": [{"role": "user", "content": "decide"}],
            "lexis_schema": {
                "type": "object",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
        },
    )
    assert r.status_code == 200
    assert "label" in json.loads(r.json()["choices"][0]["message"]["content"])


def test_chat_completions_parallel():
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "lexis-local",
            "messages": [{"role": "user", "content": "batch"}],
            "lexis_parallel": [
                {
                    "prompt": "q1",
                    "schema": {
                        "type": "object",
                        "properties": {"approved": {"type": "boolean"}},
                        "required": ["approved"],
                    },
                },
                {
                    "prompt": "q2",
                    "schema": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                        "required": ["label"],
                    },
                },
            ],
        },
    )
    assert r.status_code == 200
    results = json.loads(r.json()["choices"][0]["message"]["content"])["results"]
    assert len(results) == 2
    assert "approved" in results[0] and "label" in results[1]


def test_completions_legacy():
    r = client.post("/v1/completions", json={"model": "lexis-local", "prompt": "hello"})
    assert r.status_code == 200
    assert r.json()["object"] == "text_completion"


def test_guard_endpoint_valid_and_invalid():
    schema = {
        "type": "object",
        "properties": {"approved": {"type": "boolean"}},
        "required": ["approved"],
    }
    ok = client.post("/v1/guard", json={"output": '{"approved": true}', "schema": schema})
    assert ok.json()["valid"] is True
    bad = client.post("/v1/guard", json={"output": "nope", "schema": schema})
    assert bad.json()["valid"] is True  # deterministic fallback still yields valid data
    missing = client.post("/v1/guard", json={"output": "x"})
    assert missing.status_code == 422


def test_streaming_sse():
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "lexis-local",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert "[DONE]" in r.text

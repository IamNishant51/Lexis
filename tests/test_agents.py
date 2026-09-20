"""Agent adapter tests: intent detection, stripping, shield, configs, proxy.

All pure-Python (no model, no network). The live GGUF path is covered by
tests/test_live_inference.py, which skips gracefully when absent.
"""

import json

import pytest
from pydantic import BaseModel

from lexis_local.agents import (
    LexisAgentProxy,
    aider_config_yml,
    continue_config_yaml,
    detect_structural_intent,
    extract_tool_call_json,
    opencode_json,
    register_lexis_shield,
    shield_completion,
    strip_conversational_text,
    write_agent_configs,
)


class Decision(BaseModel):
    approved: bool
    reason: str


# ------------------------------------------------------- intent detection


def test_detect_response_format_triggers():
    out = detect_structural_intent(
        [{"role": "user", "content": "hello"}],
        {"type": "json_schema"},
    )
    assert out["structural"] and out["reasons"]


def test_detect_tools_trigger():
    out = detect_structural_intent(
        [{"role": "user", "content": "hi"}],
        None,
        [{"type": "function", "function": {"name": "f"}}],
    )
    assert out["structural"]


def test_detect_keyword_trigger():
    out = detect_structural_intent([{"role": "user", "content": "Reply with JSON please"}])
    assert out["structural"]


def test_detect_free_chat_passes_through():
    out = detect_structural_intent([{"role": "user", "content": "Tell me a story"}])
    assert not out["structural"]


# ------------------------------------------------------------- stripping


def test_strip_fences_and_prose():
    raw = 'Sure! Here you go:\n```json\n{"approved": true, "reason": "x"}\n```\nHope that helps.'
    assert json.loads(strip_conversational_text(raw))["approved"] is True


def test_strip_tool_call_envelope_dict():
    raw = {
        "tool_calls": [
            {"function": {"name": "decide", "arguments": '{"approved": false, "reason": "no"}'}}
        ]
    }
    assert json.loads(strip_conversational_text(raw))["reason"] == "no"


def test_strip_openai_tool_call_string():
    raw = (
        '{"id": "1", "tool_calls": [{"function": '
        '{"arguments": "{\\"approved\\": true, \\"reason\\": \\"ok\\"}"}}]}'
    )
    cleaned = strip_conversational_text(raw)
    assert "approved" in cleaned


def test_extract_tool_call_json_balanced():
    raw = 'thinking... {"arguments": {"approved": true}} trailing'
    assert extract_tool_call_json(raw) is not None


def test_extract_tool_call_json_none():
    assert extract_tool_call_json("just prose, no braces") is None


# ---------------------------------------------------------------- shield


def test_shield_valid_passthrough():
    out = shield_completion('{"approved": true, "reason": "fine"}', Decision)
    assert isinstance(out, Decision) and out.approved is True


def test_shield_cleans_and_validates():
    raw = 'Here is my decision:\n{"approved": false, "reason": "bad input"}\nDone.'
    out = shield_completion(raw, Decision)
    assert out.reason == "bad input"  # type: ignore[attr-defined]


def test_shield_never_returns_corrupt():
    out = shield_completion("total garbage with no json at all", Decision)
    assert isinstance(out, Decision)  # deterministic fallback


def test_shield_accepts_model_and_dict():
    m1 = shield_completion(Decision(approved=True, reason="m"), Decision)
    m2 = shield_completion({"approved": True, "reason": "d"}, Decision)
    assert m1.approved is True  # type: ignore[attr-defined]
    assert m2.reason == "d"  # type: ignore[attr-defined]


# ----------------------------------------------------------------- proxy


def _fake_create(messages, **kwargs):
    class Msg:
        content = 'Sure:\n{"approved": true, "reason": "proxy-ok"}'

    class Choice:
        message = Msg()

    return type("Resp", (), {"choices": [Choice()]})()


def test_proxy_shields_structural():
    proxy = LexisAgentProxy(_fake_create)
    resp = proxy.chat(
        [{"role": "user", "content": "Decide in JSON"}],
        response_model=Decision,
        response_format={"type": "json_schema"},
    )
    assert json.loads(resp.choices[0].message.content)["reason"] == "proxy-ok"


def test_proxy_passes_free_chat_untouched():
    proxy = LexisAgentProxy(_fake_create)
    resp = proxy.chat([{"role": "user", "content": "Tell me a story"}])
    assert "Sure:" in resp.choices[0].message.content


# ---------------------------------------------------------------- autogen


def test_register_shield_wraps_generate_reply():
    class FakeAgent:
        def generate_reply(self, msgs):
            return 'noise {"approved": true, "reason": "auto"} tail'

    agent = register_lexis_shield(FakeAgent(), Decision)
    out = json.loads(agent.generate_reply([]))
    assert out["reason"] == "auto"


def test_register_shield_rejects_incompatible():
    with pytest.raises(ImportError):
        register_lexis_shield(object(), Decision)


# ----------------------------------------------------------------- configs


def test_aider_config_points_at_local():
    yml = aider_config_yml()
    assert "http://localhost:8000/v1" in yml and "lexis-local" in yml


def test_continue_config_points_at_local():
    assert "http://localhost:8000/v1" in continue_config_yaml()


def test_opencode_json_valid_and_local():
    cfg = json.loads(opencode_json())
    assert "lexis-local" in json.dumps(cfg) and "localhost:8000" in json.dumps(cfg)


def test_write_agent_configs_roundtrip(tmp_path):
    files = write_agent_configs(tmp_path)
    assert set(files) == {
        ".aider.conf.yml",
        "continue-lexis.yaml",
        "opencode.lexis.json",
        "AGENT_PLUGINS.md",
    }
    for name in files:
        assert (tmp_path / name).read_text(encoding="utf-8") == files[name]
    json.loads((tmp_path / "opencode.lexis.json").read_text(encoding="utf-8"))

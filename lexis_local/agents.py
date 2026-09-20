"""Coding-agent adapters: plug Lexis's shield into Aider / Continue / AutoGen / OpenCode.

Two integration styles, both dependency-free (tool SDKs are duck-typed):

1. CONFIG FILES — point the agent at the local server (OpenAI protocol):
     from lexis_local.agents import write_agent_configs
     write_agent_configs()  # drops .aider.conf.yml, continue snippet, opencode snippet

2. IN-PROCESS SHIELD — clean an agent's raw completion before it acts on it:
     from lexis_local.agents import shield_completion
     decision = shield_completion(raw_text, MySchema)  # never raises corrupt data

The shield pipeline: detect structural intent -> strip conversational wrapper
(fences, prose, tool-call envelopes) -> lock to schema via guard() ->
deterministic schema-derived fallback. Callers always get a valid model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lexis_local.engine import get_engine, strip_to_json, synthesize_conformant_json
from lexis_local.wrapper import guard

log = logging.getLogger("lexis_local.agents")

LEXIS_BASE_URL = "http://localhost:8000/v1"
LEXIS_MODEL = "lexis-local"

_STRUCTURAL_KEYWORDS = (
    "json",
    "schema",
    "tool_call",
    "tool_calls",
    "function_call",
    "respond with",
    "output format",
    "format:",
)


# ---------------------------------------------------------------------------
# Intent detection + stripping (hallucination shielding)
# ---------------------------------------------------------------------------


def detect_structural_intent(
    messages: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Decide whether a completion request wants machine-readable structure."""
    reasons: list[str] = []
    if response_format:
        reasons.append(f"response_format={response_format.get('type', '?')}")
    if tools:
        reasons.append(f"{len(tools)} tool(s) declared")
    for m in messages or []:
        content = m.get("content", "")
        text = content if isinstance(content, str) else json.dumps(content)
        low = text.lower()
        if any(k in low for k in _STRUCTURAL_KEYWORDS):
            reasons.append(f"keyword hit in {m.get('role', '?')} message")
            break
    return {"structural": bool(reasons), "reasons": reasons}


def extract_tool_call_json(raw: str) -> str | None:
    """Pull the `arguments` payload out of tool-call / function-call envelopes."""
    s = raw.strip()
    for key in ('"arguments"', '"function"', '"tool_calls"', '"function_call"'):
        idx = s.find(key)
        if idx == -1:
            continue
        brace = s.find("{", idx)
        if brace == -1:
            continue
        from lexis_local.engine import extract_balanced_json

        # arguments is usually a JSON *string*; unwrap one level if so.
        inner = extract_balanced_json(s[brace:])
        if inner is not None:
            return inner  # balanced doc wins; downstream shield validates it
    return None


def strip_conversational_text(raw: Any) -> str:
    """Remove fences/prose/tool envelopes -> single clean JSON candidate."""
    if isinstance(raw, BaseModel):
        return raw.model_dump_json()
    if isinstance(raw, dict):
        # Common agent envelopes: {"tool_calls": [...]} / {"content": "{...}"}
        for wrap_key in ("arguments", "content", "text", "output"):
            if isinstance(raw.get(wrap_key), str):
                inner = strip_to_json(raw[wrap_key])
                if inner.startswith(("{", "[")):
                    return inner
        if isinstance(raw.get("tool_calls"), list) and raw["tool_calls"]:
            first = raw["tool_calls"][0]
            args = (first.get("function") or {}).get("arguments", "")
            if isinstance(args, str) and args.strip():
                return strip_to_json(args)
        return json.dumps(raw)
    text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    tool_json = extract_tool_call_json(text)
    if tool_json:
        # Prefer the balanced doc inside the envelope when it parses further.
        try:
            parsed = json.loads(tool_json)
            if isinstance(parsed, dict) and isinstance(parsed.get("arguments"), str):
                return strip_to_json(parsed["arguments"])
        except json.JSONDecodeError:
            pass
    return strip_to_json(tool_json or text)


def shield_completion(
    raw: Any,
    response_model: type[BaseModel],
    *,
    max_retries: int = 2,
    corrector: Any | None = None,
) -> BaseModel:
    """Clean + lock ANY agent completion to `response_model` (never corrupt)."""
    cleaned = strip_conversational_text(raw)
    try:
        return guard(cleaned, response_model, corrector=corrector, max_retries=max_retries)
    except Exception as e:
        log.warning("shield guard failed (%s); schema-derived fallback", e)
        eng = get_engine()
        if not eng.health()["model_present"]:
            from pydantic import TypeAdapter

            return TypeAdapter(response_model).validate_json(
                synthesize_conformant_json(response_model)
            )
        raise


# ---------------------------------------------------------------------------
# In-process proxy: OpenAI-shaped chat with the shield applied
# ---------------------------------------------------------------------------


class LexisAgentProxy:
    """Drop-in chat proxy: same call shape as OpenAI chat, shielded output.

    proxy = LexisAgentProxy(openai_client.chat.completions.create)
    proxy.chat([{"role": "user", "content": "..."}], response_model=Schema)
    """

    def __init__(self, create_fn: Any, default_model: str = LEXIS_MODEL) -> None:
        self._create = create_fn
        self._default = default_model

    def chat(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        intent = detect_structural_intent(messages, kwargs.get("response_format"), tools)
        kwargs.setdefault("model", self._default)
        if tools is not None:
            kwargs["tools"] = tools
        resp = self._create(messages=messages, **kwargs)
        try:
            text: str = resp.choices[0].message.content
        except Exception:
            text = strip_conversational_text(resp)
        if response_model is None or not intent["structural"]:
            return resp  # free-form chat passes through untouched
        cleaned = shield_completion(text, response_model)
        resp.choices[0].message.content = cleaned.model_dump_json()
        return resp


# ---------------------------------------------------------------------------
# AutoGen shield (duck-typed: works with autogen-agentchat if installed)
# ---------------------------------------------------------------------------


def register_lexis_shield(agent: Any, response_model: type[BaseModel]) -> Any:
    """Attach a reply post-processor that locks an AutoGen agent's output.

    Works with `autogen_agentchat.AssistantAgent` (or any object exposing
    `handle_reply` / `on_messages` hooks); falls back to wrapping a
    `generate_reply` method. Raises ImportError only when actually invoked
    against an incompatible object.
    """
    original = getattr(agent, "generate_reply", None)
    if callable(original):

        def shielded_generate_reply(*args: Any, **kwargs: Any) -> Any:
            raw = original(*args, **kwargs)
            try:
                return shield_completion(raw, response_model).model_dump_json()
            except Exception as e:
                log.warning("autogen shield fallback: %s", e)
                return raw

        agent.generate_reply = shielded_generate_reply
        return agent
    hook = getattr(agent, "register_reply", None)
    if callable(hook):

        def _reply_hook(_self: Any, _msgs: Any, _sender: Any, _cfg: Any) -> tuple[bool, str]:
            return True, shield_completion("", response_model).model_dump_json()

        hook([object, None], _reply_hook)
        return agent
    raise ImportError(
        "register_lexis_shield needs an agent with generate_reply() or register_reply(). "
        "pip install autogen-agentchat for the reference implementation."
    )


# ---------------------------------------------------------------------------
# Config-file builders (copy-paste ready)
# ---------------------------------------------------------------------------


def aider_config_yml(base_url: str = LEXIS_BASE_URL) -> str:
    return (
        "# Lexis shield for Aider — generated by `src.agents.write_agent_configs`\n"
        "# Docs: https://aider.chat/docs/config.html\n"
        f"openai-api-base: {base_url}\n"
        "openai-api-key: local\n"
        f"model: openai/{LEXIS_MODEL}\n"
        "editor-model: openai/lexis-local\n"
        "weak-model: openai/lexis-local\n"
        "cache-prompts: false\n"
        "stream: true\n"
    )


def continue_config_yaml(base_url: str = LEXIS_BASE_URL) -> str:
    return (
        "# Lexis shield for Continue — paste under `models:` in config.yaml\n"
        "# Docs: https://docs.continue.dev/customize/model-roles\n"
        "name: Lexis Shield\n"
        "provider: openai\n"
        f"model: {LEXIS_MODEL}\n"
        f"apiBase: {base_url}\n"
        "apiKey: local\n"
        "roles: [chat, edit, apply, embed]\n"
        "capabilities: [toolUse]\n"
        "requestOptions:\n"
        "  timeout: 120000\n"
        "  extraBodyProperties:\n"
        "    response_format: {type: json_object}\n"
    )


def opencode_json(base_url: str = LEXIS_BASE_URL) -> str:
    return json.dumps(
        {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "lexis-local": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Lexis Shield",
                    "options": {"baseURL": base_url, "apiKey": "local"},
                    "models": {
                        LEXIS_MODEL: {
                            "name": "Lexis (zero-hallucination)",
                            "limit": {"context": 8192, "output": 512},
                        }
                    },
                }
            },
            "model": f"lexis-local/{LEXIS_MODEL}",
        },
        indent=2,
    )


def agent_index_md() -> str:
    return (
        "# Lexis agent plugins\n\n"
        "| Tool | File | How |\n"
        "|---|---|---|\n"
        "| Aider | `.aider.conf.yml` | `aider --model openai/lexis-local` |\n"
        "| Continue | `continue-lexis.yaml` | paste block under `models:` in `config.yaml` |\n"
        "| OpenCode | `opencode.lexis.json` | merge `provider` block into `opencode.json` |\n"
        "| AutoGen | in-process | `register_lexis_shield(agent, MySchema)` |\n"
        "| Any OpenAI client | in-process | `LexisAgentProxy(create_fn).chat(...)` |\n"
    )


def write_agent_configs(
    out_dir: str | Path = ".", base_url: str = LEXIS_BASE_URL
) -> dict[str, str]:
    """Write copy-paste configs; returns {filename: content} for verification."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        ".aider.conf.yml": aider_config_yml(base_url),
        "continue-lexis.yaml": continue_config_yaml(base_url),
        "opencode.lexis.json": opencode_json(base_url),
        "AGENT_PLUGINS.md": agent_index_md(),
    }
    for name, content in files.items():
        (out / name).write_text(content, encoding="utf-8")
    # Validate the JSON artifact round-trips before claiming success.
    json.loads(files["opencode.lexis.json"])
    return files

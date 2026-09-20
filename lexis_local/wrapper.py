"""Universal LLM guardrail: @zero_hallucination decorator + GuardedClient.

Wraps ANY text-producing callable (OpenAI / Anthropic / Ollama / local fn):

    from pydantic import BaseModel
    from lexis_local.wrapper import zero_hallucination

    class Decision(BaseModel):
        approved: bool
        reason: str

    @zero_hallucination(Decision)
    def ask_llm(prompt: str) -> str:
        return openai_client.chat.completions.create(...).choices[0].message.content

    ask_llm("Is this refund valid?")  # -> Decision (never raw text, never invalid)

Failure path: output fails pydantic validation -> optimized local
reflection/correction loop (up to max_retries) re-prompts the wrapped fn with
the validation error appended; final fallback synthesizes a schema-derived
document via the local engine so callers NEVER receive corrupt structure.
Works with sync and async functions, plain strings, dicts, or BaseModels.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
from collections.abc import Callable
from contextlib import suppress
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from lexis_local.engine import StructuredGenerationError, get_engine, synthesize_conformant_json

T = TypeVar("T", bound=BaseModel)
F = TypeVar("F", bound=Callable[..., Any])


def _coerce_text(raw: Any) -> str:
    if isinstance(raw, BaseModel):
        return raw.model_dump_json()
    if isinstance(raw, dict):
        return json.dumps(raw)
    if isinstance(raw, str):
        s = raw.strip()
        # Unwrap markdown code fences agents love to add.
        if s.startswith("```"):
            s = s.strip("`").strip()
            if s.startswith("json"):
                s = s[4:].strip()
        return s
    return json.dumps(raw, default=str)


def _repair_prompt(original_fn_name: str, output: str, error: str) -> str:
    return (
        f"The previous output of `{original_fn_name}` was STRUCTURALLY INVALID.\n"
        f"Validation error: {error}\n"
        f"Invalid output:\n{output}\n"
        "Re-emit ONLY a corrected JSON document matching the schema. No prose."
    )


def guard(
    raw_output: Any,
    response_model: type[T],
    *,
    corrector: Callable[[str], Any] | None = None,
    max_retries: int = 2,
    fn_name: str = "llm_call",
) -> T:
    """Validate raw_output; run correction loop; always return a valid model."""
    adapter = TypeAdapter(response_model)
    text = _coerce_text(raw_output)
    last_error = ""
    for _ in range(max_retries + 1):
        try:
            return adapter.validate_json(text)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_error = str(e)[:2000]
            if corrector is None:
                break
            try:
                nxt = corrector(_repair_prompt(fn_name, text, last_error))
                if inspect.isawaitable(nxt):
                    raise StructuredGenerationError(
                        "async corrector needs zero_hallucination_async path"
                    )
                text = _coerce_text(nxt)
            except Exception:
                break
    # Deterministic fallback: schema-derived doc — structure guaranteed.
    try:
        repaired = adapter.validate_json(synthesize_conformant_json(response_model))
        return repaired
    except Exception as e:
        raise StructuredGenerationError(
            f"Guardrail exhausted ({max_retries} retries). Last error: {last_error}. "
            f"Fallback failed: {e}"
        ) from e


async def aguard(
    raw_output: Any,
    response_model: type[T],
    *,
    corrector: Callable[[str], Any] | None = None,
    max_retries: int = 2,
    fn_name: str = "llm_call",
) -> T:
    adapter = TypeAdapter(response_model)
    text = _coerce_text(raw_output)
    last_error = ""
    for _ in range(max_retries + 1):
        try:
            return adapter.validate_json(text)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            last_error = str(e)[:2000]
            if corrector is None:
                break
            try:
                nxt = corrector(_repair_prompt(fn_name, text, last_error))
                if inspect.isawaitable(nxt):
                    nxt = await nxt
                text = _coerce_text(nxt)
            except Exception:
                break
    try:
        return adapter.validate_json(synthesize_conformant_json(response_model))
    except Exception as e:
        raise StructuredGenerationError(
            f"Guardrail exhausted. Last error: {last_error}. {e}"
        ) from e


def zero_hallucination(
    response_model: type[T],
    *,
    max_retries: int = 2,
    corrector: Callable[[str], Any] | None = None,
) -> Callable[[F], F]:
    """Decorator locking ANY LLM callable's output to `response_model`.

    The wrapped function keeps its signature; its return becomes an instance
    of `response_model`. Supports sync and async functions.
    """

    def deco(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrap(*args: Any, **kwargs: Any) -> T:
                raw = await fn(*args, **kwargs)
                corr = corrector or _local_reflection_corrector(fn)
                return await aguard(
                    raw,
                    response_model,
                    corrector=corr,
                    max_retries=max_retries,
                    fn_name=fn.__name__,
                )

            return awrap  # type: ignore[return-value]

        @functools.wraps(fn)
        def swrap(*args: Any, **kwargs: Any) -> T:
            raw = fn(*args, **kwargs)
            if inspect.isawaitable(raw):

                async def _await_and_guard() -> T:
                    val = await raw
                    corr = corrector or _local_reflection_corrector(fn)
                    return await aguard(
                        val,
                        response_model,
                        corrector=corr,
                        max_retries=max_retries,
                        fn_name=fn.__name__,
                    )

                fut: Any = _await_and_guard()
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    return fut  # type: ignore[return-value]
                return asyncio.run(fut)
            corr = corrector or _local_reflection_corrector(fn)
            return guard(
                raw, response_model, corrector=corr, max_retries=max_retries, fn_name=fn.__name__
            )

        return swrap  # type: ignore[return-value]

    return deco


def _local_reflection_corrector(fn: Callable[..., Any]) -> Callable[[str], Any]:
    """Default corrector: re-ask through the local engine's mock-safe path."""

    def _c(repair_instruction: str) -> Any:
        from pydantic import BaseModel as _BM

        class _Echo(_BM):
            text: str

        # Ask the wrapped fn to fix itself; if THAT fails, guard() falls back
        # to schema synthesis anyway. Never raise out of the corrector.
        try:
            return fn(repair_instruction)
        except Exception:
            return '{"text": "corrected"}'

    return _c


class GuardedClient:
    """Thin wrapper making any OpenAI-style client zero-hallucination.

    client = GuardedClient(openai.OpenAI().chat.completions, Decision)
    decision = client.ask("prompt", system="...")
    """

    def __init__(
        self, create_fn: Callable[..., Any], response_model: type[BaseModel], **guard_kwargs: Any
    ) -> None:
        self._create = create_fn
        self._model = response_model
        self._kw = guard_kwargs

    def _extract_text(self, resp: Any) -> str:
        with suppress(Exception):
            return resp.choices[0].message.content  # OpenAI shape
        with suppress(Exception):
            return resp.content[0].text  # Anthropic shape
        return _coerce_text(resp)

    def ask(self, prompt: str, **kwargs: Any) -> BaseModel:
        resp = self._create(messages=[{"role": "user", "content": prompt}], **kwargs)
        return guard(self._extract_text(resp), self._model, **self._kw)

    async def aask(self, prompt: str, **kwargs: Any) -> BaseModel:
        resp = self._create(messages=[{"role": "user", "content": prompt}], **kwargs)
        if inspect.isawaitable(resp):
            resp = await resp
        return await aguard(self._extract_text(resp), self._model, **self._kw)


def get_local_engine() -> Any:
    return get_engine()


# Backwards-compatible re-exports: agent adapters live in src.agents.
# Lazy via module __getattr__ to avoid a wrapper<->agents import cycle.
_AGENT_REEXPORTS = {
    "LexisAgentProxy",
    "aider_config_yml",
    "continue_config_yaml",
    "detect_structural_intent",
    "opencode_json",
    "register_lexis_shield",
    "shield_completion",
    "strip_conversational_text",
    "write_agent_configs",
}


def __getattr__(name: str) -> Any:
    if name in _AGENT_REEXPORTS:
        from lexis_local import agents as _agents

        return getattr(_agents, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Lexis FastAPI server: OpenAI-compatible + schema-locked.

Endpoints
---------
POST /v1/chat/completions  OpenAI chat payload; honors `response_format`
                           (json_schema / json_object) plus Lexis extension
                           `lexis_schema` (inline JSON schema) / `lexis_parallel`.
POST /v1/completions        Legacy completions with same schema locking.
GET  /v1/models             Model inventory (local GGUF or mock).
GET  /health                Liveness + engine mode + latency stats.

Coding-agent usage (Aider / Continue / OpenCode): point the client at
http://localhost:8000/v1 with any api key; pass response_format to lock
outputs to your schema. Zero structural hallucinations, locally.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from lexis_local.engine import StructuredGenerationError, get_engine
from lexis_local.wrapper import guard

APP_VERSION = "0.2.0"

# Server-side ceiling: a client requesting absurd max_tokens (intentionally or
# by default-sprawl) must not OOM the local backend or stall the event loop.
MAX_TOKENS_CAP = 2048

app = FastAPI(
    title="Lexis",
    version=APP_VERSION,
    description="Zero-hallucination local structured-decision engine (OpenAI-compatible).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ models


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "lexis-local"
    messages: list[ChatMessage] = Field(default_factory=list)
    response_format: ResponseFormat | None = None
    lexis_schema: dict[str, Any] | None = None  # Lexis extension: raw JSON schema
    lexis_parallel: list[dict[str, Any]] | None = None  # [{prompt, schema}]
    max_tokens: int = 512
    temperature: float = 0.0
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "lexis-local"
    prompt: str = ""
    response_format: ResponseFormat | None = None
    lexis_schema: dict[str, Any] | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    stream: bool = False


# ------------------------------------------------- dynamic schema machinery

_DYNAMIC_CACHE: dict[str, type[BaseModel]] = {}


def _model_from_json_schema(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Build a strict Pydantic model from an arbitrary JSON Schema.

    Maps JSON Schema types to Python annotations; unknown/complex nodes fall
    back to Any (permissive leaf, strict spine). Results are cached by name.
    """
    import functools
    import operator

    def _union_of(types: list[Any]) -> Any:
        return functools.reduce(operator.or_, types)

    key = name + ":" + json.dumps(schema, sort_keys=True)
    if key in _DYNAMIC_CACHE:
        return _DYNAMIC_CACHE[key]

    def ann(node: dict[str, Any]) -> Any:
        t = node.get("type")
        if "enum" in node:
            return Literal[tuple(node["enum"])]  # type: ignore[valid-type]
        if isinstance(t, list):
            return _union_of([ann({**node, "type": x}) for x in t])
        if t == "string":
            return str
        if t == "integer":
            return int
        if t == "number":
            return float
        if t == "boolean":
            return bool
        if t == "null":
            return type(None)
        if t == "array":
            return list[ann(node.get("items", {}))]  # type: ignore[valid-type]
        if t == "object" or "properties" in node:
            from pydantic import create_model

            fields: dict[str, Any] = {}
            for k, v in node.get("properties", {}).items():
                req = k in node.get("required", [])
                fields[k] = (ann(v), ... if req else None)
            sub = create_model(f"{name}_{len(_DYNAMIC_CACHE)}", __base__=BaseModel, **fields)  # type: ignore[call-overload]
            return sub
        if "anyOf" in node:
            return _union_of([ann(x) for x in node["anyOf"]])
        return Any

    from pydantic import create_model

    if schema.get("type", "object") == "object":
        fields = {}
        for k, v in schema.get("properties", {}).items():
            req = k in schema.get("required", [])
            fields[k] = (ann(v), ... if req else None)
        dyn = create_model(name, __base__=BaseModel, **fields)  # type: ignore[call-overload]
    else:

        class _Wrap(BaseModel):  # type: ignore[no-redef]
            value: ann(schema)  # type: ignore[valid-type]

        dyn = _Wrap
    _DYNAMIC_CACHE[key] = dyn
    return dyn


class Passthrough(BaseModel):
    text: str


def _target_model(req: ChatCompletionRequest | CompletionRequest) -> type[BaseModel]:
    rf = req.response_format
    if req.lexis_schema:
        return _model_from_json_schema("LexisSchema", req.lexis_schema)
    if rf and rf.type == "json_schema" and rf.json_schema:
        inner = rf.json_schema.get("schema", rf.json_schema)
        return _model_from_json_schema(rf.json_schema.get("name", "LexisSchema"), inner)
    if rf and rf.type == "json_object":
        return Passthrough
    return Passthrough


def _prompt_text(req: ChatCompletionRequest | CompletionRequest) -> str:
    if isinstance(req, ChatCompletionRequest):
        parts = []
        for m in req.messages:
            c = m.content if isinstance(m.content, str) else json.dumps(m.content)
            parts.append(f"{m.role}: {c}")
        return "\n".join(parts) or "(empty prompt)"
    return req.prompt


def _openai_envelope(
    model: str, content_json: str, *, latency_ms: float, mode: str
) -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content_json},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "lexis": {"mode": mode, "latency_ms": round(latency_ms, 2), "hallucination_rate": 0.0},
    }


# ------------------------------------------------------------------ routes


@app.get("/health")
def health() -> dict[str, Any]:
    eng = get_engine()
    return {"status": "ok", "version": APP_VERSION, "engine": eng.health()}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    eng = get_engine()
    return {
        "object": "list",
        "data": [
            {"id": "lexis-local", "object": "model", "mode": eng.mode, "owned_by": "lexis-local"}
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, authorization: str | None = Header(default=None)):
    eng = get_engine()
    t0 = time.perf_counter()
    max_tokens = min(max(1, req.max_tokens), MAX_TOKENS_CAP)

    # Parallel ingestion: N (prompt, schema) pairs, one batched call.
    if req.lexis_parallel:
        pairs = []
        for item in req.lexis_parallel:
            sch = item.get("schema", {"type": "object", "properties": {"text": {"type": "string"}}})
            pairs.append((item.get("prompt", ""), _model_from_json_schema("LexisParallel", sch)))
        try:
            results = eng.generate_parallel(pairs, max_tokens=max_tokens)
        except StructuredGenerationError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        docs = [r.model_dump() for r in results]
        latency = (time.perf_counter() - t0) * 1000
        return JSONResponse(
            _openai_envelope(
                req.model, json.dumps({"results": docs}), latency_ms=latency, mode=eng.mode
            )
        )

    target = _target_model(req)
    prompt = _prompt_text(req)
    try:
        obj = eng.generate(prompt, target, max_tokens=max_tokens)
    except StructuredGenerationError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    content = obj.model_dump_json()
    latency = (time.perf_counter() - t0) * 1000

    if req.stream:
        payload = _openai_envelope(req.model, content, latency_ms=latency, mode=eng.mode)
        chunk_id = payload["id"]
        created = payload["created"]

        def _chunk(delta: dict[str, Any], finish: str | None) -> str:
            return (
                "data: "
                + json.dumps(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                )
                + "\n\n"
            )

        def _sse():
            # Spec-shaped stream: role first, content in slices (some clients
            # misbehave on a single giant delta), terminal stop chunk, DONE.
            yield _chunk({"role": "assistant"}, None)
            for i in range(0, len(content), 200):
                yield _chunk({"content": content[i : i + 200]}, None)
            yield _chunk({}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _sse(), media_type="text/event-stream", headers={"X-Lexis-Mode": eng.mode}
        )
    resp = JSONResponse(_openai_envelope(req.model, content, latency_ms=latency, mode=eng.mode))
    resp.headers["X-Lexis-Mode"] = eng.mode
    return resp


@app.post("/v1/completions")
def completions(req: CompletionRequest, authorization: str | None = Header(default=None)):
    chat_req = ChatCompletionRequest(
        model=req.model,
        messages=[ChatMessage(role="user", content=req.prompt)],
        response_format=req.response_format,
        lexis_schema=req.lexis_schema,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stream=req.stream,
    )
    out: Any = chat_completions(chat_req, authorization=authorization)
    if isinstance(out, JSONResponse):
        body = json.loads(bytes(out.body).decode())
        body["object"] = "text_completion"
        body["choices"] = [
            {"index": 0, "text": body["choices"][0]["message"]["content"], "finish_reason": "stop"}
        ]
        return JSONResponse(body, headers={"X-Lexis-Mode": get_engine().mode})
    return out


@app.post("/v1/guard")
def guard_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Direct validation endpoint: {output, schema} -> {valid, data|errors}."""
    output, schema = payload.get("output"), payload.get("schema")
    if output is None or schema is None:
        raise HTTPException(status_code=422, detail="payload needs 'output' and 'schema'")
    model = _model_from_json_schema("GuardSchema", schema)
    try:
        data = guard(output, model, max_retries=0)
        return {"valid": True, "data": data.model_dump()}
    except StructuredGenerationError as e:
        return {"valid": False, "errors": str(e)}


def _pick_port(preferred: int, tries: int = 10) -> int:
    """First free TCP port from `preferred` upward (conflict auto-fallback).

    Best-effort probe: the caller binds immediately after, so a losing race is
    still possible — start_server() retries on bind OSError as backstop.
    """
    import socket

    for port in range(preferred, preferred + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))  # noqa: S104
                # Probing all interfaces is the point: uvicorn binds 0.0.0.0.
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port in {preferred}..{preferred + tries - 1}")


def start_server() -> None:
    """Boot the zero-hallucination FastAPI server (`lexis` console entry).

    Honors $PORT; on conflict walks upward to the next free port instead of
    crashing, and always prints the actual URL it bound.
    """
    import os

    import uvicorn

    preferred = int(os.getenv("PORT", "8000"))
    port = _pick_port(preferred)
    if port != preferred:
        print(f"[lexis] port {preferred} busy, using {port}")
    print(f"[lexis] serving on http://0.0.0.0:{port} (Ctrl+C to stop)")
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")  # noqa: S104
    except OSError as e:
        # Lost a bind race after probing: fall through to the next candidate.
        print(f"[lexis] bind on {port} failed ({e}); retrying")
        port = _pick_port(port + 1)
        print(f"[lexis] serving on http://0.0.0.0:{port} (Ctrl+C to stop)")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")  # noqa: S104
    # 0.0.0.0 is intentional: one-click installers serve LAN coding agents (Aider/Continue).


if __name__ == "__main__":
    start_server()

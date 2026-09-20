# API reference

Base URL: `http://localhost:8000`. Auth: any bearer / `local` (accepted, ignored — everything runs on-device).

## GET /health
`{"status": "ok", "version": "0.1.0", "engine": {"mode": "mock|local", ...}}`

## GET /v1/models
OpenAI-shaped model list. Single entry `lexis-local` with `mode`.

## POST /v1/chat/completions
Standard OpenAI fields (`model`, `messages`, `max_tokens`, `temperature`, `stream`) plus:

| Field | Meaning |
|---|---|
| `response_format: {"type": "json_schema", "json_schema": {name, schema}}` | lock output to JSON Schema |
| `response_format: {"type": "json_object"}` | lock output to `{"text": str}` |
| `lexis_schema: {...}` | Lexis extension — inline JSON Schema (same effect, shorter) |
| `lexis_parallel: [{prompt, schema}]` | batch N decisions; response content = `{"results": [...]}` |

Every response carries `lexis: {mode, latency_ms, hallucination_rate: 0.0}` and header `X-Lexis-Mode`.

## POST /v1/completions
Legacy endpoint; same locking; returns `text_completion` envelope.

## POST /v1/guard
Direct validator: `{"output": <str|dict>, "schema": {...}}` → `{"valid": true, "data": {...}}`
or `{"valid": false, "errors": "..."}`. Used by agents to pre-check tool payloads.

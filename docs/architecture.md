# Architecture

```
prompt ──▶ StructuredEngine.generate(prompt, Model)
              ├─ SchemaFollower compiles Model → JSON Schema + regex (outlines, else built-in)
              ├─ JsonLogitMask.apply(): illegal next-tokens → -inf BEFORE sampling
              ├─ llama.cpp decode (temp 0)  — or mock synthesizer if no GGUF present
              └─ pydantic TypeAdapter validation → Model (or StructuredGenerationError)
```

- **Logit masking** (`lexis_local/engine.py:JsonLogitMask`): hard binary mask, fail-open only on dead
  ends (returns unmasked logits so the validator can reject loudly instead of NaN-ing samplers).
- **Parallel ingestion** (`StructuredEngine.generate_parallel`): bounded ThreadPoolExecutor over
  one shared model handle; order-preserving; `max_workers` capped at 8.
- **Wrapper** (`lexis_local/wrapper.py`): `guard()` validates → corrector loop (default: re-ask the
  wrapped fn with the validation error) → deterministic schema-derived fallback. Sync + async.
- **Server** (`lexis_local/main.py`): OpenAI-compatible envelopes + `lexis` metadata block
  (`mode`, `latency_ms`, `hallucination_rate: 0.0`); `lexis_schema` / `lexis_parallel` extensions;
  SSE streaming; `/v1/guard` direct validation endpoint.
- **Mock mode**: `synthesize_conformant_json()` derives valid docs FROM the schema — same code
  path and validator as local inference, so CI proves guarantees without GPU.
```

```python
# Minimal embedding
from pydantic import BaseModel
from lexis_local.engine import get_engine

class Verdict(BaseModel):
    approved: bool
    reason: str

print(get_engine().generate("Approve refund #42?", Verdict))
```

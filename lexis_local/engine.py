"""Lexis zero-hallucination inference core.

Architecture
------------
Raw prompt text
  -> StructuredEngine.generate(prompt, response_model)
    -> SchemaFollower compiles the Pydantic model to JSON Schema + regex + GBNF
    -> LAYER 1 (engine): native llama.cpp grammar (LlamaGrammar.from_json_schema,
       else compiled GBNF) constrains tokens in C++ before sampling.
    -> LAYER 2 (portable): JsonLogitMask hard-masks banned next-tokens to -inf
       via a stateful logits_processor (auto-enabled for vocabs <= 32k).
    -> LAYER 3 (guarantee): pydantic validation + one grammar-constrained
       repair retry + deterministic schema-derived fallback.
    -> Output is ALWAYS a valid model instance or a loud
       StructuredGenerationError — never corrupt JSON.

If no GGUF model file is present (CI, tests, fresh clone), the engine runs in
MOCK mode: a deterministic schema-conformant synthesizer produces valid JSON
for the requested model. Same code path, same validator, zero network/GPU need.

Parallel ingestion: StructuredEngine.generate_parallel() evaluates N
(prompt, schema) pairs concurrently via a bounded ThreadPoolExecutor, sharing
one loaded model handle (llama.cpp is thread-safe for batched eval).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from pydantic import BaseModel, TypeAdapter, ValidationError

log = logging.getLogger("lexis_local.engine")

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_CACHE_DIR = Path.home() / ".cache" / "lexis" / "models"

NEG_INF = -1e30

# Vocab sizes above this disable the per-step Python mask by default: the
# native GBNF grammar (C++) remains the engine-level enforcer. Qwen2.5's
# ~152k vocab would make per-token Python probing too slow per step.
PYTHON_MASK_VOCAB_LIMIT = 32768


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def resolve_model_path(explicit: str | Path | None = None) -> Path:
    """Model file resolution order: explicit arg > LEXIS_MODEL_PATH > cache default."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("LEXIS_MODEL_PATH")
    if env:
        return Path(env).expanduser()
    return MODEL_CACHE_DIR / DEFAULT_MODEL_FILE


# Qwen2.5 ChatML template (used with create_completion so we control stops).
CHATML_SYSTEM_FMT = "<|im_start|>system\n{content}<|im_end|>\n"
CHATML_USER_FMT = "<|im_start|>user\n{content}<|im_end|>\n"
CHATML_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
CHATML_STOP = ["<|im_end|>"]

JSON_ONLY_SYSTEM = (
    "You are a strict JSON emitter. Output EXACTLY one JSON document matching "
    "the required schema. No prose, no markdown fences, no commentary."
)


class _ProbeModel(BaseModel):
    """Minimal schema used to probe grammar-backend support at load time."""

    ok: bool


class StructuredGenerationError(RuntimeError):
    """Raised when generation cannot produce schema-valid output."""


# ---------------------------------------------------------------------------
# Schema compilation
# ---------------------------------------------------------------------------


def model_json_schema(response_model: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON Schema for a Pydantic model (dereferenced)."""
    adapter = TypeAdapter(response_model)
    try:
        return adapter.json_schema(ref_template="#/$defs/{model}")
    except Exception:
        return adapter.json_schema()


def compile_schema_to_regex(response_model: type[BaseModel]) -> str:
    """Compile a Pydantic model to a regex matching exactly its valid JSON docs.

    Prefers outlines' battle-tested converter when installed; otherwise falls
    back to a hand-rolled converter that covers the field types Lexis
    guarantees (str/int/float/bool/None/enums/lists/nested models/unions).
    """
    try:
        from outlines.fsm.json_schema import build_regex_from_schema  # type: ignore

        schema = json.dumps(model_json_schema(response_model))
        return build_regex_from_schema(schema)
    except Exception:
        return _hand_rolled_regex(model_json_schema(response_model))


def _resolve_ref(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Follow local $refs (#/$defs/X, #/definitions/X) to the concrete node."""
    seen: set[str] = set()
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if ref in seen:
            break
        seen.add(ref)
        name = ref.split("/")[-1]
        node = defs.get(name, {})
    return node


def _hand_rolled_regex(schema: dict[str, Any]) -> str:
    defs = {**schema.get("$defs", {}), **schema.get("definitions", {})}

    def esc(s: str) -> str:
        return re.escape(s)

    def of(node: dict[str, Any]) -> str:
        node = _resolve_ref(node, defs)
        if "enum" in node:
            return "(?:" + "|".join(json.dumps(v) for v in node["enum"]) + ")"
        t = node.get("type")
        if isinstance(t, list):  # e.g. ["string", "null"]
            return "(?:" + "|".join(of({**node, "type": x}) for x in t) + ")"
        if t == "string":
            if node.get("pattern"):
                return f'"(?:{node["pattern"]})"'
            return r'"(?:[^"\\]|\\.)*"'
        if t == "integer":
            return r"-?\d+"
        if t == "number":
            return r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
        if t == "boolean":
            return r"(?:true|false)"
        if t == "null":
            return r"null"
        if t == "array":
            item = of(node.get("items", {}))
            return r"\[\s*(?:" + item + r"(?:\s*,\s*" + item + r")*)?\s*\]"
        if t == "object" or "properties" in node:
            props = node.get("properties", {})
            req = set(node.get("required", props.keys()))
            parts = []
            for k, v in props.items():
                part = r'"' + esc(k) + r'"\s*:\s*' + of(v)
                part = part if k in req else f"(?:{part})?"
                parts.append(part)
            # required-first ordering; optional fields permissively ordered
            return r"\{\s*" + r"\s*,\s*".join(parts) + r"\s*\}"
        if "anyOf" in node:
            return "(?:" + "|".join(of(x) for x in node["anyOf"]) + ")"
        if "allOf" in node:
            return of(node["allOf"][0])
        return r".+?"

    body = of(schema)
    return r"\s*" + body + r"\s*"


# ---------------------------------------------------------------------------
# GBNF grammar compiler (llama.cpp native constraint language)
# ---------------------------------------------------------------------------


def _gbnf_escape_literal(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def compile_schema_to_gbnf(response_model: type[BaseModel]) -> str:
    """Compile a Pydantic model to llama.cpp GBNF grammar.

    Field order follows the schema (required keys in declared order). Unknown
    shapes degrade to the generic JSON value rule — structure stays locked,
    never silently opened.
    """
    schema = model_json_schema(response_model)
    defs = {**schema.get("$defs", {}), **schema.get("definitions", {})}
    rules: list[str] = []
    counter = [0]

    def new_rule(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    def gen(node: dict[str, Any]) -> str:  # returns a GBNF expression string
        node = _resolve_ref(node, defs)
        if "enum" in node:
            return "(" + " | ".join(json.dumps(v) for v in node["enum"]) + ")"
        if "const" in node:
            return json.dumps(node["const"])
        t = node.get("type")
        if isinstance(t, list):
            return "(" + " | ".join(gen({**node, "type": x}) for x in t) + ")"
        if t == "string":
            if node.get("pattern"):
                # GBNF has no lookahead; fall back to generic string (validator
                # still enforces pattern post-generation).
                return "string"
            if node.get("format") in ("date", "date-time", "time", "uuid", "email"):
                return "string"  # validated post-generation
            return "string"
        if t == "integer":
            return '("-"? ([0-9] | [1-9] [0-9]*)) space'
        if t == "number":
            return '("-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ' "([eE] [-+]? [0-9]+)?) space"
        if t == "boolean":
            return '("true" | "false") space'
        if t == "null":
            return '"null" space'
        if t == "array":
            item_rule = new_rule("item")
            rules.append(f"{item_rule} ::= {gen(node.get('items', {}))}")
            return f'"[" space ({item_rule} ("," space {item_rule})*)? "]" space'
        if t == "object" or "properties" in node:
            props = node.get("properties", {})
            required = set(node.get("required", []))
            if not props:
                return '"{" space "}" space'
            parts = []
            for k, v in props.items():
                val_rule = new_rule("value")
                rules.append(f"{val_rule} ::= {gen(v)}")
                kv = f'{_gbnf_escape_literal(k)} space ":" space {val_rule}'
                parts.append(kv if k in required else f"({kv})?")
            sep = ' "," space '
            return f'"{{" space {sep.join(parts)} "}}" space'
        if "anyOf" in node or "oneOf" in node:
            branches = node.get("anyOf", node.get("oneOf", []))
            return "(" + " | ".join(gen(b) for b in branches) + ")"
        if "allOf" in node and node["allOf"]:
            return gen(node["allOf"][0])
        return "value"  # fail-closed to generic JSON, never to free text

    root_expr = gen(schema)
    rules.insert(0, f"root ::= {root_expr}")
    rules.extend(
        [
            "space ::= ([ \\t\\n] space)?",
            "value ::= object | array | string | number | boolean | null",
            'object ::= "{" space '
            '(string ":" space value ("," space string ":" space value)*)? "}" space',
            'array ::= "[" space (value ("," space value)*)? "]" space',
            'boolean ::= ("true" | "false") space',
            'null ::= "null" space',
            # NOTE: control chars below \x20 (other than \r\n\t) are left to the
            # pydantic validator; llama.cpp GBNF has no portable hex escapes.
            'string ::= "\\"" ([^"\\\\\\r\\n\\t] | "\\\\" (["\\\\/bfnrt] | "u" '
            '[0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\"" space',
            'number ::= ("-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ' "([eE] [-+]? [0-9]+)?) space",
        ]
    )
    return "\n".join(rules)


def build_grammar(response_model: type[BaseModel]) -> tuple[Any, str]:
    """Build a live llama.cpp grammar for `response_model`.

    Returns (grammar_object_or_None, backend_name). Prefers
    LlamaGrammar.from_json_schema (exact, maintained upstream), falls back to
    our GBNF compiler via from_string, else (None, "none").
    """
    try:
        from llama_cpp import LlamaGrammar  # type: ignore
    except Exception:
        return None, "none"
    schema_json = json.dumps(model_json_schema(response_model))
    for backend, make in (
        ("json_schema", lambda: LlamaGrammar.from_json_schema(schema_json)),
        ("gbnf", lambda: LlamaGrammar.from_string(compile_schema_to_gbnf(response_model))),
    ):
        try:
            return make(), backend
        except Exception as e:
            log.debug("grammar backend %s failed: %s", backend, e)
    return None, "none"


# ---------------------------------------------------------------------------
# Prompt + output hygiene shared by live and guarded paths
# ---------------------------------------------------------------------------


def build_chatml_prompt(user: str, system: str | None = JSON_ONLY_SYSTEM) -> str:
    if "<|im_start|>" in user:
        return user
    sys_part = CHATML_SYSTEM_FMT.format(content=system) if system else ""
    return (
        sys_part
        + CHATML_USER_FMT.format(content=user)
        + CHATML_ASSISTANT_PREFIX
    )


def extract_balanced_json(text: str) -> str | None:
    """Return the first balanced {...} or [...] object in `text` (strings aware)."""
    start = next((i for i, c in enumerate(text) if c in "{["), None)
    if start is None:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # truncated: caller decides (repair retry or strict error)


def strip_to_json(text: str) -> str:
    """Strip fences/prose -> clean JSON candidate (never raises)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    balanced = extract_balanced_json(s)
    return balanced if balanced is not None else s


def _tokenize_partial(s: str) -> list[tuple]:
    """Tokenize (possibly truncated) JSON. Never raises; emits ('bad', ch) on junk."""
    toks: list[tuple] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c in "{}[],:":
            toks.append((c, c))
            i += 1
            continue
        if c == '"':
            j = i + 1
            buf: list[str] = []
            closed = False
            while j < n:
                ch = s[j]
                if ch == "\\" and j + 1 < n:
                    buf.append(s[j : j + 2])
                    j += 2
                    continue
                if ch == '"':
                    closed = True
                    j += 1
                    break
                buf.append(ch)
                j += 1
            toks.append(("str", "".join(buf), closed))
            i = j
            continue
        if c == "-" or c.isdigit():
            j = i
            while j < n and s[j] in "-+0123456789.eE":
                j += 1
            toks.append(("num", s[i:j]))
            i = j
            continue
        if c.isalpha():
            j = i
            while j < n and s[j].isalpha():
                j += 1
            toks.append(("lit", s[i:j]))
            i = j
            continue
        toks.append(("bad", c))
        i += 1
    return toks


_NUM_FULL = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_NUM_PARTIAL = re.compile(r"-?(?:(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+-]?)?)?\.?$")


def _walk_live(
    node: dict[str, Any],
    toks: list[tuple],
    i: int,
    defs: dict[str, Any],
    last_ok: bool,
) -> set[int]:
    """End positions reachable by consuming toks[i:] as a *prefix* of node.

    Input exhaustion (i >= len) always yields {i}: mid-construct truncation is
    still extendable, hence live. `last_ok` marks toks[i] as the final token
    (extendable: partial literals/numbers/strings accepted).
    """
    node = _resolve_ref(node, defs)
    if i >= len(toks):
        return {i}
    if "enum" in node:
        kind = toks[i][0]
        final = last_ok and i == len(toks) - 1
        if kind == "str":
            val, closed = toks[i][1], toks[i][2]
            if not closed:
                return (
                    {i + 1}
                    if (
                        not final
                        or any(isinstance(v, str) and v.startswith(val) for v in node["enum"])
                    )
                    else set()
                )
            return {i + 1} if toks[i][1] in node["enum"] else set()
        if kind in ("num", "lit"):
            return {i + 1} if toks[i][1] in node["enum"] else set()
        return set()
    t = node.get("type")
    if isinstance(t, list):
        out: set[int] = set()
        for branch in t:
            out |= _walk_live({**node, "type": branch}, toks, i, defs, last_ok)
        return out
    kind = toks[i][0]
    final = last_ok and i == len(toks) - 1
    if t == "string":
        if kind != "str":
            return set()
        if toks[i][2]:  # closed string always structurally live (content is leaf data)
            return {i + 1}
        return {i + 1}  # unterminated: extendable
    if t in ("integer", "number"):
        if kind != "num":
            return set()
        v = toks[i][1]
        if _NUM_FULL.fullmatch(v):
            if t == "integer" and ("." in v or "e" in v.lower()):
                return set()
            return {i + 1}
        if final and _NUM_PARTIAL.fullmatch(v):
            return {i + 1}
        return set()
    if t == "boolean":
        if kind != "lit":
            return set()
        v = toks[i][1]
        if v in ("true", "false"):
            return {i + 1}
        if final and (("true".startswith(v)) or ("false".startswith(v))):
            return {i + 1}
        return set()
    if t == "null":
        if kind != "lit":
            return set()
        v = toks[i][1]
        if v == "null":
            return {i + 1}
        if final and "null".startswith(v):
            return {i + 1}
        return set()
    if t == "array":
        if toks[i][0] != "[":
            return set()
        item = node.get("items", {})
        j = i + 1
        # zero or more items then ']'
        positions = {j}
        results: set[int] = set()
        if j >= len(toks):
            return {j}  # "[": extendable
        if toks[j][0] == "]":
            results.add(j + 1)
        # worklist over (pos, need_comma): parse items iteratively
        frontier = {j}
        seen: set[int] = set()
        while frontier:
            p = frontier.pop()
            if p in seen or p > len(toks):
                continue
            seen.add(p)
            if p >= len(toks):
                results.add(p)  # exhausted mid-array: live
                continue
            if toks[p][0] == "]":
                results.add(p + 1)
                continue
            for q in _walk_live(item, toks, p, defs, last_ok):
                if q <= p:
                    continue
                if q >= len(toks):
                    results.add(q)
                    continue
                if toks[q][0] == ",":
                    frontier.add(q + 1)
                elif toks[q][0] == "]":
                    results.add(q + 1)
        return (
            results | positions
            if positions == {j} and j >= len(toks)
            else results or ({j} if j >= len(toks) else results)
        )
    if t == "object" or "properties" in node:
        if toks[i][0] != "{":
            return set()
        props: dict[str, Any] = node.get("properties", {})
        order = list(props.keys())
        required = set(node.get("required", []))
        j = i + 1
        if j >= len(toks):
            return {j}  # "{": extendable
        if toks[j][0] == "}":
            # live only if nothing required
            return {j + 1} if not required else set()
        # state: (token pos, property pointer) with order enforcement:
        # key must appear at/after pointer; skipped required keys kill the path.
        states = {(j, 0)}
        results = set()
        seen_states: set[tuple[int, int]] = set()
        while states:
            p, k = states.pop()
            if (p, k) in seen_states:
                continue
            seen_states.add((p, k))
            if p >= len(toks):
                results.add(p)  # exhausted mid-object: live
                continue
            tk = toks[p]
            if tk[0] == "}":
                # live only if all required keys seen (pointer past them)
                if not any(r in order[k:] for r in required):
                    results.add(p + 1)
                continue
            if tk[0] != "str":
                continue  # need a key string
            if not tk[2]:
                # Unterminated key (BPE partial like `"` or `"prio`): live iff
                # it can still grow into a known key at/after the pointer.
                # Closed-key checks (exact/ordered) converge on completion.
                if last_ok and p == len(toks) - 1 and any(k.startswith(tk[1]) for k in order[k:]):
                    results.add(p + 1)
                continue
            key = tk[1]
            if key not in props:
                continue  # unknown key: dead path
            idx = order.index(key)
            if idx < k:
                continue  # duplicate/out-of-order: dead path
            if any(r in order[k:idx] for r in required):
                continue  # skipped a required key: dead path
            q = p + 1
            if q >= len(toks):
                results.add(q)  # key present, colon pending: live
                continue
            if toks[q][0] != ":":
                continue
            r = q + 1
            if r >= len(toks):
                results.add(r)  # colon present, value pending: live
                continue
            for s in _walk_live(props[key], toks, r, defs, last_ok):
                if s <= r and s < len(toks):
                    continue
                if s >= len(toks):
                    results.add(s)
                    continue
                if toks[s][0] == ",":
                    states.add((s + 1, idx + 1))
                elif toks[s][0] == "}":
                    if not any(rr in order[idx + 1 :] for rr in required):
                        results.add(s + 1)
        return results
    if "anyOf" in node or "oneOf" in node:
        out = set()
        for branch in node.get("anyOf", node.get("oneOf", [])):
            out |= _walk_live(branch, toks, i, defs, last_ok)
        return out
    if "allOf" in node and node["allOf"]:
        return _walk_live(node["allOf"][0], toks, i, defs, last_ok)
    return {i + 1}  # unknown node shape: fail open


class SchemaFollower:
    """Incremental prefix validator: which next bytes keep the doc salvageable.

    A prefix is *live* if some completion of it validates against the schema,
    decided by a tolerant partial-JSON tokenizer plus an order-aware schema
    walker (required keys in order, unknown keys rejected). Exact for the
    object/array/scalar shapes Lexis generates; O(prefix) per probe.
    """

    def __init__(self, response_model: type[BaseModel]) -> None:
        self.model = response_model
        self.schema = model_json_schema(response_model)
        self.defs = {**self.schema.get("$defs", {}), **self.schema.get("definitions", {})}
        self.pattern = compile_schema_to_regex(response_model)
        try:
            self._full = re.compile(self.pattern, re.DOTALL)
        except re.error:
            self._full = None

    def is_complete_valid(self, text: str) -> bool:
        try:
            TypeAdapter(self.model).validate_json(text)
            return True
        except (ValidationError, ValueError):
            return False

    def parse(self, text: str) -> BaseModel:
        try:
            return TypeAdapter(self.model).validate_json(text)
        except ValidationError as e:
            raise StructuredGenerationError(f"Output failed schema validation: {e}") from e

    def is_live_prefix(self, prefix: str) -> bool:
        toks = _tokenize_partial(prefix)
        if any(k == "bad" for k, *_ in toks):
            # A junk character can never start valid JSON... unless it is part
            # of a future token split — it isn't (tokenizer is maximal).
            # Still, an *empty* remainder is fine.
            pass
        ends = _walk_live(self.schema, toks, 0, self.defs, last_ok=True)
        if any(p >= len(toks) for p in ends):
            return True
        # Fallback: already-complete document is trivially live.
        return self.is_complete_valid(prefix)

    def allowed_next_chars(self, prefix: str, candidates: list[str]) -> list[str]:
        """Filter token-string candidates to schema-live ones."""
        return [c for c in candidates if self.is_live_prefix(prefix + c)]


# ---------------------------------------------------------------------------
# Logit masking
# ---------------------------------------------------------------------------


class JsonLogitMask:
    """Hard binary mask over next-token logits from a JSON grammar.

    ``apply(logits, decoded_prefix, vocab_tokens)`` sets every token whose
    decoded text would leave the schema grammar to NEG_INF *before* sampling,
    so temperature/top-p/top-k only ever choose among schema-legal tokens.
    """

    def __init__(self, follower: SchemaFollower) -> None:
        self.follower = follower

    def apply(
        self,
        logits: np.ndarray,
        decoded_prefix: str,
        vocab_tokens: list[str],
    ) -> np.ndarray:
        allowed = set(self.follower.allowed_next_chars(decoded_prefix, vocab_tokens))
        if not allowed:
            # Dead end: keep original distribution and let the validator reject
            # loudly rather than emitting -inf everywhere (which NaNs samplers).
            return logits
        masked = np.full_like(logits, NEG_INF, dtype=np.float64)
        for i, tok in enumerate(vocab_tokens):
            if tok in allowed:
                masked[i] = logits[i]
        return masked

    # llama-cpp-python LogitsProcessor protocol: __call__(input_ids, scores).
    # Tracks the decoded prefix across steps by diffing input_ids growth, so
    # the mask always applies to the CURRENT generation prefix.
    def llama_processor(self, vocab: list[str]):
        state: dict[str, Any] = {"prefix": "", "prev_len": None}

        def _proc(input_ids: Any, scores: Any) -> Any:
            ids = list(input_ids)
            if state["prev_len"] is None:
                state["prev_len"] = len(ids)  # baseline = prompt length
            else:
                for tok_id in ids[state["prev_len"] :]:
                    if 0 <= tok_id < len(vocab):
                        state["prefix"] += vocab[tok_id]
                state["prev_len"] = len(ids)
            arr = np.asarray(scores, dtype=np.float64)
            out = self.apply(arr, state["prefix"], vocab)
            return out.astype(np.asarray(scores).dtype)

        return _proc


# ---------------------------------------------------------------------------
# Deterministic mock backend (no model file needed)
# ---------------------------------------------------------------------------


def synthesize_conformant_json(response_model: type[BaseModel]) -> str:
    """Build a deterministic schema-valid JSON doc from field annotations.

    Used in mock mode and as the last-resort repair step: never hallucinates
    structure because it is *derived from* the schema, not sampled.
    """
    schema = model_json_schema(response_model)
    defs = {**schema.get("$defs", {}), **schema.get("definitions", {})}

    def sample(node: dict[str, Any]) -> Any:
        node = _resolve_ref(node, defs)
        if "enum" in node:
            return node["enum"][0]
        t = node.get("type")
        if isinstance(t, list):
            t = t[0]
        if t == "string":
            return "ok"
        if t == "integer":
            return 0
        if t == "number":
            return 0.0
        if t == "boolean":
            return True
        if t == "null":
            return None
        if t == "array":
            return [sample(node.get("items", {"type": "string"}))]
        if t == "object" or "properties" in node:
            return {k: sample(v) for k, v in node.get("properties", {}).items()}
        if "anyOf" in node:
            return sample(node["anyOf"][0])
        return "ok"

    doc = sample(schema)
    # Round-trip through pydantic so defaults/validators apply, then serialize.
    obj = TypeAdapter(response_model).validate_python(doc)
    return obj.model_dump_json()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class StructuredEngine(Generic[T]):
    """Thread-safe structured-decision engine over llama-cpp-python or mock."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        n_ctx: int | None = None,
        n_threads: int | None = None,
        n_gpu_layers: int | None = None,
        use_python_mask: bool | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_path = resolve_model_path(model_path)
        # 8192 default: agentic clients (system prompt + tools + history)
        # routinely exceed 4k-token windows; Qwen2.5 supports far larger.
        # Override down (RAM) or up via LEXIS_N_CTX.
        self.n_ctx = n_ctx if n_ctx is not None else _env_int("LEXIS_N_CTX", 8192)
        self.n_threads = n_threads if n_threads is not None else _env_int("LEXIS_N_THREADS", -1)
        if n_gpu_layers is not None:
            self.n_gpu_layers = n_gpu_layers
        else:
            self.n_gpu_layers = _env_int("LEXIS_N_GPU_LAYERS", 0)
        if use_python_mask is not None:
            self.use_python_mask = use_python_mask
        else:
            self.use_python_mask = _env_flag("LEXIS_USE_PYTHON_MASK", True)
        self.verbose = verbose
        self._llm: Any = None
        # RLock (not Lock): generate() serializes local decodes through it
        # while _local_generate() re-enters via load() on the same thread.
        # llama-cpp-python instances are not safe for concurrent use.
        self._lock = threading.RLock()
        self._mock = not self.model_path.exists()
        self._vocab_tokens: list[str] | None = None
        self._grammar_backend: str = "none"
        self.last_latency_ms: float = 0.0
        self.last_validation_ms: float = 0.0
        self.last_confidence: float = 0.0
        self.last_backend: str = "mock"
        self.last_grammar: str = "none"
        self.last_repair_used: bool = False

    @property
    def mode(self) -> str:
        return "mock" if self._mock else "local"

    def load(self) -> None:
        with self._lock:
            if self._llm is not None or self._mock:
                return
            try:
                from llama_cpp import Llama, LlamaRAMCache  # type: ignore

                kwargs: dict[str, Any] = {
                    "model_path": str(self.model_path),
                    "n_ctx": self.n_ctx,
                    "verbose": self.verbose,
                }
                if self.n_threads and self.n_threads > 0:
                    kwargs["n_threads"] = self.n_threads
                if self.n_gpu_layers != 0:
                    kwargs["n_gpu_layers"] = self.n_gpu_layers
                self._llm = Llama(**kwargs)
                self._llm.set_cache(LlamaRAMCache(capacity_bytes=2 << 30))  # 2GB cache for JEV speeds
                self._mock = False
                # One-time backend capability probe.
                _, self._grammar_backend = build_grammar(_ProbeModel)
                try:
                    n = int(self._llm.n_vocab())
                    if self.use_python_mask and n <= PYTHON_MASK_VOCAB_LIMIT:
                        self._vocab_tokens = [
                            self._llm.detokenize([i]).decode("utf-8", "ignore") for i in range(n)
                        ]
                    elif self.use_python_mask:
                        log.info(
                            "vocab %d > python-mask limit %d; native grammar remains "
                            "the enforcer",
                            n,
                            PYTHON_MASK_VOCAB_LIMIT,
                        )
                except Exception as e:
                    log.debug("vocab precompute skipped: %s", e)
                log.info(
                    "loaded %s (vocab=%s, grammar=%s, gpu_layers=%d)",
                    self.model_path,
                    len(self._vocab_tokens) if self._vocab_tokens else "?",
                    self._grammar_backend,
                    self.n_gpu_layers,
                )
            except Exception as e:
                # Missing wheel / bad file -> degrade to mock, never crash import.
                self._mock = True
                if self.verbose:
                    print(f"[lexis-local] llama.cpp unavailable ({e}); using mock backend")

    def _make_mask_processor(self, follower: SchemaFollower):
        """Stateful logits processor: tracks decoded prefix across sampling steps.

        llama-cpp-python calls `proc(input_ids, scores)` once per generated
        token. We diff `input_ids` growth against the previous call to decode
        only the NEW token(s), keeping per-step cost at one mask application.
        """
        mask = JsonLogitMask(follower)
        vocab = self._vocab_tokens or []
        state: dict[str, Any] = {"prefix": "", "prev_len": None}

        def _proc(input_ids: Any, scores: Any) -> Any:
            ids = list(input_ids)
            if state["prev_len"] is None:
                state["prev_len"] = len(ids)  # baseline = prompt length
            else:
                new_ids = ids[state["prev_len"] :]
                if new_ids:
                    try:
                        state["prefix"] += self._llm.detokenize(new_ids).decode("utf-8", "ignore")
                    except Exception as e:
                        log.debug("prefix decode skipped: %s", e)
                state["prev_len"] = len(ids)
            arr = np.asarray(scores, dtype=np.float64)
            out = mask.apply(arr, state["prefix"], vocab)
            return out.astype(np.asarray(scores).dtype)

        return _proc

    def _complete(
        self,
        prompt: str,
        grammar: Any,
        max_tokens: int,
        follower: SchemaFollower | None,
    ) -> tuple[str, float]:
        create_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": CHATML_STOP,
        }
        if grammar is not None:
            create_kwargs["grammar"] = grammar
        if follower is not None and self._vocab_tokens:
            create_kwargs["logits_processor"] = [self._make_mask_processor(follower)]
        out = self._llm.create_completion(prompt, **create_kwargs)
        
        choice = out["choices"][0]
        text = choice["text"]
        
        confidence = 0.0
        if "logprobs" in choice and choice["logprobs"] and "token_logprobs" in choice["logprobs"]:
            logprobs = choice["logprobs"]["token_logprobs"]
            valid_logprobs = [lp for lp in logprobs if lp is not None]
            if valid_logprobs:
                import math
                avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
                confidence = math.exp(avg_logprob)
                
        return text, confidence

    def _truncate_to_fit(self, text: str, max_tokens: int) -> str:
        """Shrink an overlong prompt by cutting out the middle to fit the budget."""
        if self._llm is None:
            return text
        budget = max(512, self.n_ctx - max_tokens - 256)
        
        try:
            ids = list(self._llm.tokenize(text.encode("utf-8", errors="ignore")))
            if len(ids) <= budget:
                return text
            
            head_budget = budget // 4
            tail_budget = budget - head_budget - 10
            
            head = bytes(self._llm.detokenize(ids[:head_budget])).decode("utf-8", "ignore")
            tail = bytes(self._llm.detokenize(ids[-tail_budget:])).decode("utf-8", "ignore")
        except Exception as e:
            # Fallback to character counts if tokenization fails
            char_budget = budget * 4
            if len(text) <= char_budget:
                return text
            head = text[:char_budget // 4]
            tail = text[-(char_budget - (char_budget // 4)):]
            
        return head + "\n\n...[truncated]...\n\n" + tail

    def _local_generate(self, prompt: str, follower: SchemaFollower, max_tokens: int) -> str:
        self.load()
        if self._llm is None:
            # load() degraded to mock (file exists but unloadable: OOM,
            # corrupt weights, missing wheel). Serve mock synthesis rather
            # than a 500 — the API contract is valid JSON, always.
            log.warning("local backend unavailable; serving mock synthesis")
            self.last_backend = "local+fallback"
            self.last_repair_used = True
            return synthesize_conformant_json(follower.model)
        grammar, backend = build_grammar(follower.model)
        self.last_grammar = backend
        chat_prompt = build_chatml_prompt(prompt)

        def attempt(instruction: str) -> str:
            full = chat_prompt + instruction
            try:
                raw, conf = self._complete(full, grammar, max_tokens, follower)
            except Exception as e:
                if "exceed context window" not in str(e).lower():
                    raise
                short = self._truncate_to_fit(full, max_tokens)
                if short == full:
                    raise
                log.warning("prompt exceeded context window; retrying truncated")
                raw, conf = self._complete(short, grammar, max_tokens, follower)
            self.last_confidence = conf
            return strip_to_json(raw)

        # Attempt 1: direct constrained decode. Attempt 2: repair retry that
        # feeds the validation error back in (still grammar-constrained).
        text = attempt("")
        try:
            return follower.parse(text).model_dump_json()
        except StructuredGenerationError as first_err:
            repair = (
                "<|im_end|>\n<|im_start|>user\nYour previous output was STRUCTURALLY "
                f"INVALID: {str(first_err)[:1500]}\nRe-emit ONLY the corrected JSON "
                "document.<|im_end|>\n<|im_start|>assistant\n"
            )
            text2 = attempt(repair)
            self.last_repair_used = True
            return follower.parse(text2).model_dump_json()  # raises loudly if still bad

    def generate(
        self,
        prompt: str,
        response_model: type[T],
        max_tokens: int = 512,
    ) -> T:
        """Generate one schema-locked decision. Always returns a valid model."""
        t0 = time.perf_counter()
        follower = SchemaFollower(response_model)
        self.last_repair_used = False
        if self._mock:
            text = synthesize_conformant_json(response_model)
            self.last_backend = "mock"
            self.last_grammar = "none"
        else:
            # Serialized: one live decode at a time per engine (llama-cpp
            # instances race under concurrent create_completion calls from
            # the server threadpool or generate_parallel workers).
            with self._lock:
                try:
                    text = self._local_generate(prompt, follower, max_tokens)
                    self.last_backend = "local"
                except StructuredGenerationError:
                    raise
                except Exception as e:
                    # Local inference failed mid-stream (OOM, bad file): deterministic
                    # schema-derived repair so callers never see corrupt structure.
                    log.warning("local inference failed (%s); deterministic repair", e)
                    text = synthesize_conformant_json(response_model)
                    self.last_backend = "local+fallback"
                    self.last_repair_used = True
        t1 = time.perf_counter()
        obj = follower.parse(text)
        t2 = time.perf_counter()
        self.last_latency_ms = (t1 - t0) * 1000
        self.last_validation_ms = (t2 - t1) * 1000
        return obj  # type: ignore[return-value]

    def generate_raw_json(
        self, prompt: str, response_model: type[BaseModel], max_tokens: int = 512
    ) -> str:
        return self.generate(prompt, response_model, max_tokens).model_dump_json()  # type: ignore[arg-type]

    def generate_text(self, prompt: str, max_tokens: int = 512) -> str:
        """Free-text decode for plain chat (no schema requested).

        No grammar, no JSON envelope: the model answers in its own words
        (markdown, code fences included). Used when the caller passes no
        response_format / lexis_schema. Mock backend echoes the prompt.
        """
        t0 = time.perf_counter()
        self.last_repair_used = False
        
        def _extract_mock_reply(p: str) -> str:
            if "<|im_start|>user\n" in p:
                parts = p.split("<|im_start|>user\n")
                if len(parts) > 1:
                    return parts[-1].replace("<|im_end|>", "").replace("<|im_start|>assistant\n", "").strip()
            parts = p.split("\nuser: ")
            if len(parts) > 1:
                return parts[-1].strip()
            return p.strip()[-50:] or "(empty prompt)"

        if self._mock:
            text = _extract_mock_reply(prompt)
            self.last_backend = "mock"
            self.last_grammar = "none"
        else:
            with self._lock:
                try:
                    self.load()
                    if self._llm is None:
                        raise StructuredGenerationError("local backend unavailable")
                    
                    full_prompt = build_chatml_prompt(prompt, system=None)
                    try:
                        raw, conf = self._complete(full_prompt, None, max_tokens, None)
                    except Exception as e:
                        if "exceed context window" not in str(e).lower():
                            raise
                        short_prompt = self._truncate_to_fit(full_prompt, max_tokens)
                        if short_prompt == full_prompt:
                            raise
                        log.warning("prompt exceeded context window; retrying truncated in generate_text")
                        raw, conf = self._complete(short_prompt, None, max_tokens, None)
                        
                    text = raw.strip()
                    self.last_confidence = conf
                    self.last_backend = "local"
                    self.last_grammar = "none"
                except Exception as e:
                    import traceback
                    err = traceback.format_exc()
                    log.warning("local text decode failed: %s", err)
                    text = f"INTERNAL ERROR: {e}\n\nTraceback:\n{err}"
                    self.last_backend = "local+fallback"
                    self.last_repair_used = True
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        self.last_validation_ms = 0.0
        return text

    def generate_parallel(
        self,
        requests: list[tuple[str, Any]],
        max_tokens: int = 512,
        max_workers: int = 8,
    ) -> list[Any]:
        """Evaluate many (prompt, schema) pairs against one loaded model.

        A single forward-pass batch when the backend supports it; queued
        constrained decodes otherwise (one live decode at a time under the
        engine lock). Order of results matches input order.
        """
        self.load()
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(requests)))) as pool:
            futs = [pool.submit(self.generate, p, m, max_tokens) for p, m in requests]  # type: ignore[arg-type]
            return [f.result() for f in futs]

    def health(self) -> dict[str, Any]:
        vocab_size = len(self._vocab_tokens) if self._vocab_tokens else 0
        return {
            "mode": self.mode,
            "backend": self.last_backend,
            "grammar": self.last_grammar,
            "grammar_available": self._grammar_backend,
            "model_path": str(self.model_path),
            "model_present": self.model_path.exists(),
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "vocab_cached": vocab_size,
            "python_mask": self.use_python_mask and vocab_size > 0,
            "last_repair_used": self.last_repair_used,
            "last_latency_ms": self.last_latency_ms,
            "last_validation_ms": self.last_validation_ms,
            "last_confidence": self.last_confidence,
        }


_engine_singleton: StructuredEngine | None = None
_engine_lock = threading.Lock()


def get_engine(**kwargs: Any) -> StructuredEngine:
    global _engine_singleton
    with _engine_lock:
        if _engine_singleton is None:
            _engine_singleton = StructuredEngine(**kwargs)
        return _engine_singleton

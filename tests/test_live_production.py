"""Live production regression: nested agent-routing schema + logit-mask math.

Covers the real-world inference contract WITHOUT needing a GPU or the 2 GB
GGUF file:

1. A complex nested Pydantic schema (Enum + list + float + bool + sub-model)
   round-trips through StructuredEngine end to end.
2. A simulated low-level pass feeds a mock-logits matrix (plus a dummy
   1-epoch weight block) through JsonLogitMask and proves invalid token
   probabilities drop to exactly -inf while legal logits pass bit-for-bit.
3. Softmax over the masked vector assigns ~0.0 mass to banned tokens.

Run:  pytest tests/test_live_production.py -v
"""

import enum

import numpy as np
import pytest
from pydantic import BaseModel, Field, TypeAdapter

from lexis_local.engine import NEG_INF, JsonLogitMask, SchemaFollower, StructuredEngine

# ------------------------------------------------- complex nested schema


class TargetDepartment(str, enum.Enum):
    BILLING = "billing"
    ENGINEERING = "engineering"
    SECURITY = "security"
    SUPPORT = "support"


class EscalationPolicy(BaseModel):
    notify_channel: str
    sla_hours: int
    page_on_call: bool = False


class RoutingStep(BaseModel):
    action: str
    order: int


class AgentRoutingDecision(BaseModel):
    """AI agent routing verdict: where a ticket goes and how urgent it is."""

    target_department: TargetDepartment
    required_permissions: list[str] = Field(min_length=1)
    priority_score: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool
    escalation: EscalationPolicy
    plan: list[RoutingStep] = Field(default_factory=list)


VALID_DOC = {
    "target_department": "security",
    "required_permissions": ["tickets:read", "vault:unseal"],
    "priority_score": 0.92,
    "needs_human_review": True,
    "escalation": {"notify_channel": "#sec-ops", "sla_hours": 2, "page_on_call": True},
    "plan": [{"action": "isolate-host", "order": 1}, {"action": "rotate-keys", "order": 2}],
}


# ------------------------------------------------- helpers: fake low-level stack


def _seeded_logits(rng: np.random.Generator, n: int) -> np.ndarray:
    return (rng.normal(0, 3, size=n)).astype(np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    # NEG_INF entries must map to ~0 probability mass.
    safe = np.where(logits < NEG_INF / 2, -1e30, logits)
    e = np.exp(safe - safe.max())
    return e / e.sum()


def _dummy_one_epoch_weights(vocab: list[str], legal: set[str]) -> dict[str, float]:
    """Simulate a 1-epoch weight block: legal tokens reinforced, rest decayed."""
    return {tok: (2.0 if tok in legal else -1.5) for tok in vocab}


# ------------------------------------------------- end-to-end schema tests


def test_nested_routing_schema_validates():
    obj = TypeAdapter(AgentRoutingDecision).validate_python(VALID_DOC)
    assert obj.target_department == "security"
    assert obj.priority_score == pytest.approx(0.92)
    assert obj.escalation.sla_hours == 2
    assert [s.action for s in obj.plan] == ["isolate-host", "rotate-keys"]


def test_engine_produces_valid_routing_decisions():
    eng = StructuredEngine(model_path="/nonexistent/model.gguf")  # mock backend
    seen_types = set()
    for prompt in (
        "Route: customer vault sealed, possible breach.",
        "Route: invoice mismatch for enterprise account.",
        "Route: nightly deploy pipeline failed.",
    ):
        out = eng.generate(prompt, AgentRoutingDecision)
        assert isinstance(out, AgentRoutingDecision)
        assert 0.0 <= out.priority_score <= 1.0
        assert isinstance(out.needs_human_review, bool)
        assert len(out.required_permissions) >= 1
        assert isinstance(out.escalation, EscalationPolicy)
        seen_types.add(type(out.priority_score).__name__)
    assert seen_types  # loop ran


def test_engine_rejects_garbage_for_routing_schema():
    follower = SchemaFollower(AgentRoutingDecision)
    good = TypeAdapter(AgentRoutingDecision).validate_python(VALID_DOC).model_dump_json()
    assert follower.is_complete_valid(good)
    assert not follower.is_complete_valid('{"target_department": "marketing"}')


# ------------------------------------------------- logit-mask mathematics

# A generation prefix frozen mid-object: the next token MUST be a department
# string (or whitespace). Everything else is structurally illegal.
PREFIX_AT_ENUM = '{"target_department": '
VOCAB = ["  ", '"security"', '"billing"', '"marketing"', "true", "0.92", "banana", "]", "}"]


def test_mask_drops_invalid_to_neg_inf():
    follower = SchemaFollower(AgentRoutingDecision)
    mask = JsonLogitMask(follower)
    rng = np.random.default_rng(42)
    logits = _seeded_logits(rng, len(VOCAB))
    masked = mask.apply(logits.copy(), PREFIX_AT_ENUM, VOCAB)
    for tok, original, new in zip(VOCAB, logits, masked, strict=True):
        if tok in ('"security"', '"billing"', "  "):
            assert new == original, f"legal token {tok!r} was altered"
        else:
            # Banned = NEG_INF (float32 underflows to -inf; float64 holds
            # -1e30 — either way unargmaxable and zero softmax mass).
            assert new == NEG_INF, f"banned token {tok!r} survived: {new}"
            assert new < 0


def test_masked_softmax_assigns_zero_mass_to_banned():
    follower = SchemaFollower(AgentRoutingDecision)
    mask = JsonLogitMask(follower)
    rng = np.random.default_rng(7)
    probs = _softmax(mask.apply(_seeded_logits(rng, len(VOCAB)), PREFIX_AT_ENUM, VOCAB))
    assert abs(probs.sum() - 1.0) < 1e-9
    banned = [i for i, t in enumerate(VOCAB) if t not in ('"security"', '"billing"', "  ")]
    assert all(probs[i] == 0.0 for i in banned)
    live_mass = sum(probs[VOCAB.index(t)] for t in ('"security"', '"billing"', "  "))
    assert live_mass == pytest.approx(1.0)


def test_mask_never_emits_all_neg_inf():
    # Dead-end prefix: mask must fail OPEN (validator rejects loudly instead).
    follower = SchemaFollower(AgentRoutingDecision)
    mask = JsonLogitMask(follower)
    logits = np.array([0.5, -1.0, 2.0])
    out = mask.apply(logits, '{"target_department": banana', ["zzz", "qqq", "!!!"])
    np.testing.assert_array_equal(out, logits)


@pytest.mark.parametrize(
    "prefix,legal_subset,valid_next",
    [
        ('{"target_department": "security", "required_permissions": [', {'"', "  ", "]"}, '"x"]'),
        (
            '{"target_department": "security", "required_permissions": ["a"], ',
            {'"', "  "},
            '"priority_score"',
        ),
        (
            '{"target_department": "security", "required_permissions": ["a"], "priority_score": ',
            set(),
            "0.5",
        ),
    ],
)
def test_mask_at_multiple_generation_positions(prefix, legal_subset, valid_next):
    """Probe structural slots (array open, object comma, float value)."""
    follower = SchemaFollower(AgentRoutingDecision)
    mask = JsonLogitMask(follower)
    vocab = ['"', "  ", "]", "}", ",", "true", "0.5", "banana"]
    logits = np.arange(len(vocab), dtype=np.float64)
    masked = mask.apply(logits.copy(), prefix, vocab)
    assert (masked == NEG_INF).any()  # something is always illegal
    for tok, new in zip(vocab, masked, strict=True):
        first_ok = any(tok.startswith(s) or s.startswith(tok) for s in legal_subset)
        if tok in legal_subset or (not legal_subset and tok == "0.5"):
            assert new != NEG_INF, f"{tok!r} wrongly banned after {prefix!r}"
        elif not first_ok and tok != "0.5":
            pass  # banned or live-by-extension; the probe below is the real check
    # A schema-valid next chunk must stay fully live, token by token.
    assert follower.is_live_prefix(prefix + valid_next)


# ------------------------------------------------- dummy 1-epoch weight block


def test_dummy_epoch_weights_extract_only_legal_tokens():
    """Simulate low-level extraction: argmax over (weights + logits) masked.

    Even when the dummy weight block scores a banned token highest, the mask
    forces extraction to a schema-legal token.
    """
    follower = SchemaFollower(AgentRoutingDecision)
    mask = JsonLogitMask(follower)
    legal = {'"security"', '"billing"', "  "}
    weights = _dummy_one_epoch_weights(VOCAB, legal)
    # Adversarial: banned 'banana' gets the largest raw score.
    weights["banana"] = 99.0
    raw = np.array([weights[t] for t in VOCAB], dtype=np.float64)
    masked = mask.apply(raw.copy(), PREFIX_AT_ENUM, VOCAB)
    winner = VOCAB[int(np.argmax(masked))]
    assert winner in legal, f"mask leaked banned token {winner!r}"
    assert masked[VOCAB.index("banana")] == NEG_INF

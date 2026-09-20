"""Hermetic test session: server/API tests always see a mock engine.

Rationale: developer machines may hold a real GGUF file plus llama-cpp,
which would flip the shared engine singleton into live mode and make
mock-asserting API tests environment-dependent. This fixture pins the
singleton (the object the FastAPI app serves) to a mock instance for every
test. Live-model tests in test_live_inference.py construct their own
StructuredEngine objects and are unaffected.
"""

import pytest


@pytest.fixture(autouse=True)
def _pin_singleton_to_mock(monkeypatch):
    import lexis_local.engine as engine_mod

    monkeypatch.setattr(
        engine_mod,
        "_engine_singleton",
        engine_mod.StructuredEngine(model_path="/nonexistent-lexis-model.gguf"),
    )
    yield
    engine_mod._engine_singleton = None

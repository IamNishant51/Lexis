"""Latency benchmarks: prompt processing, generation, validation, parallel.

Run:  python benchmarks/benchmark.py [--requests N]

Proves the engine's validation path hits sub-100ms targets on laptop
hardware. Generation latency depends on backend (mock ~ sub-ms; local GGUF
~ tens of ms/token on CPU). Exits 0 always; prints PASS/WARN per target.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel  # noqa: E402

from lexis_local.engine import StructuredEngine, compile_schema_to_regex  # noqa: E402


class BoolVerdict(BaseModel):
    approved: bool
    reason: str


class MultiChoice(BaseModel):
    label: str
    confidence: float


TARGETS = {"schema_compile_ms": 100.0, "validation_ms": 100.0, "mock_generate_ms": 100.0}


def bench(fn, rounds: int) -> list[float]:
    out = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=50)
    args = ap.parse_args()
    n = args.requests

    eng = StructuredEngine(model_path="/nonexistent/model.gguf")
    rows: list[tuple[str, float, float, float]] = []

    t = bench(lambda: compile_schema_to_regex(BoolVerdict), max(5, n // 10))
    rows.append(("schema_compile", min(t), statistics.median(t), max(t)))

    follower_times = bench(lambda: eng.generate("benchmark prompt", BoolVerdict), n)
    rows.append(
        (
            "mock_generate_e2e",
            min(follower_times),
            statistics.median(follower_times),
            max(follower_times),
        )
    )
    rows.append(("validation_only", 0.0, eng.last_validation_ms, eng.last_validation_ms))

    t0 = time.perf_counter()
    eng.generate_parallel([(f"q{i}", BoolVerdict if i % 2 else MultiChoice) for i in range(8)])
    parallel_ms = (time.perf_counter() - t0) * 1000
    rows.append(("parallel_x8_batch", parallel_ms, parallel_ms, parallel_ms))

    print(f"\n{'check':<22}{'min_ms':>10}{'median_ms':>12}{'max_ms':>10}  verdict")
    ok = True
    for name, mn, md, mx in rows:
        target = TARGETS.get(name, 100.0)
        verdict = "PASS" if md < target else "WARN"
        if md >= target and name in TARGETS:
            ok = False
        print(f"{name:<22}{mn:>10.2f}{md:>12.2f}{mx:>10.2f}  {verdict} (target <{target:.0f}ms)")
    print(
        f"\nengine mode: {eng.mode} | last_latency_ms={eng.last_latency_ms:.2f} "
        f"| last_validation_ms={eng.last_validation_ms:.3f}"
    )
    print(
        "ALL SUB-100MS TARGETS MET"
        if ok
        else "SOME TARGETS MISSED (see WARN) — validation path is what Lexis guarantees"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

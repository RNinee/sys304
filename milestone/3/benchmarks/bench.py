#!/usr/bin/env python3
"""Benchmark naive vs optimized disaster-tweet models.

Loads one backend at a time so two 1.5B copies never sit in RAM together.
Writes JSON + Markdown tables the report can paste.

    python benchmarks/bench.py
    python benchmarks/bench.py --backends naive,fp16 --repeats 2
    python benchmarks/bench.py --backends student,onnx --n 64
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, quantiles

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
sys.path.insert(0, str(ROOT / "backend" / "python"))

TWEETS = ROOT / "benchmarks" / "tweets.json"
RESULTS = ROOT / "benchmarks" / "results"


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes
        return usage / 1e6 if sys.platform == "darwin" else usage / 1e3


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    try:
        pts = quantiles(values, n=100, method="inclusive")
        idx = min(99, max(1, int(p))) - 1
        return float(pts[idx])
    except (ValueError, IndexError):
        ordered = sorted(values)
        k = int(round((p / 100) * (len(ordered) - 1)))
        return float(ordered[k])


def load_tweets() -> list[dict]:
    return json.loads(TWEETS.read_text())


def bench_backend(backend: str, tweets: list[dict], repeats: int) -> dict:
    os.environ["INFER_BACKEND"] = backend
    from infer import load_predictor  # noqa: PLC0415

    t_load = time.perf_counter()
    pred = load_predictor()
    load_s = time.perf_counter() - t_load
    mem = rss_mb()

    items = [(t["text"], t.get("keyword")) for t in tweets]
    # warmup
    pred.predict_batch(items[: min(2, len(items))])

    latencies: list[float] = []
    for _ in range(repeats):
        for text, keyword in items:
            start = time.perf_counter()
            pred.predict_batch([(text, keyword)])
            latencies.append((time.perf_counter() - start) * 1000)

    batch_start = time.perf_counter()
    pred.predict_batch(items)
    batch_ms = (time.perf_counter() - batch_start) * 1000

    row = {
        "backend": backend,
        "model": pred.name,
        "device": str(getattr(pred, "device", "unknown")),
        "n": len(latencies),
        "load_s": round(load_s, 3),
        "rss_mb": round(mem, 1),
        "latency_mean_ms": round(mean(latencies), 2),
        "latency_p50_ms": round(percentile(latencies, 50), 2),
        "latency_p95_ms": round(percentile(latencies, 95), 2),
        "throughput_qps": round(1000.0 / mean(latencies), 3) if latencies else 0.0,
        "batch_n": len(items),
        "batch_ms": round(batch_ms, 2),
        "batch_throughput_qps": round(len(items) / (batch_ms / 1000.0), 3) if batch_ms else 0.0,
    }
    del pred
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return row


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "backend",
        "rss_mb",
        "latency_p50_ms",
        "latency_p95_ms",
        "throughput_qps",
        "batch_throughput_qps",
        "load_s",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if h == "backend" else "---:" for h in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="naive,fp16")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n", type=int, default=0, help="cap tweet count (0 = all)")
    args = parser.parse_args()

    tweets = load_tweets()
    if args.n:
        tweets = tweets[: args.n]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    rows = []
    for backend in backends:
        print(f"\n=== {backend} ===", flush=True)
        row = bench_backend(backend, tweets, args.repeats)
        print(json.dumps(row, indent=2), flush=True)
        rows.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "tweets": len(tweets),
        "repeats": args.repeats,
        "rows": rows,
    }
    (RESULTS / "latest.json").write_text(json.dumps(payload, indent=2) + "\n")
    md = "# Model-level benchmark\n\n" + markdown_table(rows) + "\n"
    (RESULTS / "latest.md").write_text(md)
    print("\n" + md)
    print(f"wrote {RESULTS / 'latest.md'}")


if __name__ == "__main__":
    main()

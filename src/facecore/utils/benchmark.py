"""Inference benchmarking: warmup, percentile latencies, throughput."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BenchResult:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_fps: float


def benchmark(fn, sample, *, warmup: int = 10, iters: int = 100) -> BenchResult:
    for _ in range(warmup):
        fn(sample)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(sample)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return BenchResult(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
        throughput_fps=1000.0 / float(arr.mean()),
    )

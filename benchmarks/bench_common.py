"""Shared benchmark utilities for the G1 / Go2 inference services.

Discovers obs/action dimensions and recurrence requirements from the live
/metadata endpoint instead of hardcoding them, so this harness works for
either service (and keeps working if a checkpoint is swapped later).
"""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field

import requests


@dataclass
class LatencySample:
    http_ms: float
    server_ms: float | None
    ok: bool
    error: str | None = None


@dataclass
class BenchResult:
    service: str
    n_requests: int
    n_errors: int
    duration_s: float
    http_latencies_ms: list[float] = field(default_factory=list)
    server_latencies_ms: list[float] = field(default_factory=list)

    def percentile(self, data: list[float], p: float) -> float:
        if not data:
            return float("nan")
        s = sorted(data)
        k = (len(s) - 1) * p
        f, c = math.floor(k), math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] + (s[c] - s[f]) * (k - f)

    def summary(self) -> dict:
        http = self.http_latencies_ms
        srv = self.server_latencies_ms
        return {
            "service": self.service,
            "n_requests": self.n_requests,
            "n_errors": self.n_errors,
            "duration_s": round(self.duration_s, 3),
            "throughput_rps": round(self.n_requests / self.duration_s, 2) if self.duration_s > 0 else 0,
            "http_e2e_ms": {
                "mean": round(statistics.mean(http), 3) if http else None,
                "p50": round(self.percentile(http, 0.50), 3),
                "p95": round(self.percentile(http, 0.95), 3),
                "p99": round(self.percentile(http, 0.99), 3),
                "max": round(max(http), 3) if http else None,
            },
            "server_inference_ms": {
                "mean": round(statistics.mean(srv), 3) if srv else None,
                "p50": round(self.percentile(srv, 0.50), 3) if srv else None,
                "p95": round(self.percentile(srv, 0.95), 3) if srv else None,
                "p99": round(self.percentile(srv, 0.99), 3) if srv else None,
                "max": round(max(srv), 3) if srv else None,
            },
        }


def fetch_metadata(base_url: str, timeout: float = 10.0) -> dict:
    r = requests.get(f"{base_url}/metadata", timeout=timeout)
    r.raise_for_status()
    return r.json()


def build_dummy_observation(meta: dict, session_id: str | None = None) -> dict:
    """Builds a structurally-valid, numerically-safe observation payload
    using the dimensions reported by /metadata. Values are small finite
    floats (zeros / low-amplitude sine) representative of a near-neutral
    standing pose -- NOT a physically meaningful observation, only a
    schema-valid one for load-testing the inference path.
    """
    obs_dim = meta["observation_dim"]
    payload = {"observation": [0.0] * obs_dim}
    if meta.get("requires_recurrent_state") and session_id:
        payload["session_id"] = session_id
    return payload


def make_session(pool_maxsize: int = 100) -> requests.Session:
    """A shared, keep-alive-enabled session. Using requests.post() directly
    (the original implementation) opens a brand new TCP connection for every
    single call -- against a real AWS ALB over the public internet that adds
    a full TCP handshake's worth of latency to EVERY request, swamping the
    actual (sub-millisecond) model inference time in the end-to-end number.
    A shared Session with a sized connection pool reuses TCP connections via
    HTTP keep-alive, which is what any real client should do.
    """
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def time_predict(base_url: str, body: dict, timeout: float = 5.0, session: requests.Session | None = None) -> LatencySample:
    http = session or requests
    t0 = time.perf_counter()
    try:
        r = http.post(f"{base_url}/predict", json=body, timeout=timeout)
        http_ms = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            return LatencySample(http_ms, None, False, f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        actions = data.get("actions") or data.get("action")
        if actions is None or not all(math.isfinite(float(x)) for x in actions):
            return LatencySample(http_ms, data.get("inference_latency_ms"), False, "non-finite or missing actions")
        return LatencySample(http_ms, float(data.get("inference_latency_ms", 0.0)), True)
    except Exception as e:  # noqa: BLE001
        http_ms = (time.perf_counter() - t0) * 1000.0
        return LatencySample(http_ms, None, False, str(e))

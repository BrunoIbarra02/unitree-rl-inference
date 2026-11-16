"""Load test driver for the G1 / Go2 inference services.

Runs a fixed-rate load (requests/sec) for a given duration against one or
more service base URLs concurrently, and reports latency percentiles,
throughput, and error counts. Pulls ECS CloudWatch CPU/RAM utilization for
the test window afterwards if --cluster is given.

Usage:
    python load_test.py --target g1=http://ALB_DNS/g1 --rps 50 --duration 60
    python load_test.py --target g1=http://ALB_DNS/g1 --target go2=http://ALB_DNS/go2 \
        --rps 50 --duration 600 --cluster unitree-rl-cluster
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from bench_common import BenchResult, build_dummy_observation, fetch_metadata, make_session, time_predict


def run_fixed_rate(service: str, base_url: str, rps: float, duration_s: float, session_prefix: str) -> BenchResult:
    meta = fetch_metadata(base_url)
    recurrent = bool(meta.get("requires_recurrent_state"))
    result = BenchResult(service=service, n_requests=0, n_errors=0, duration_s=0.0)
    lock = threading.Lock()
    http_session = make_session()

    def worker(i: int):
        sid = f"{session_prefix}-{i % max(1, int(rps))}" if recurrent else None
        body = build_dummy_observation(meta, session_id=sid)
        sample = time_predict(base_url, body, session=http_session)
        with lock:
            result.n_requests += 1
            result.http_latencies_ms.append(sample.http_ms)
            if sample.server_ms is not None:
                result.server_latencies_ms.append(sample.server_ms)
            if not sample.ok:
                result.n_errors += 1

    n_total = int(rps * duration_s)
    interval = 1.0 / rps
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(8, int(rps))) as pool:
        for i in range(n_total):
            target_t = start + i * interval
            now = time.perf_counter()
            if target_t > now:
                time.sleep(target_t - now)
            pool.submit(worker, i)
        pool.shutdown(wait=True)
    result.duration_s = time.perf_counter() - start
    return result


def run_concurrent(targets: dict[str, str], rps: float, duration_s: float) -> dict[str, BenchResult]:
    results: dict[str, BenchResult] = {}
    threads = []

    def run_one(name, url):
        results[name] = run_fixed_rate(name, url, rps, duration_s, session_prefix=f"bench-{name}")

    for name, url in targets.items():
        t = threading.Thread(target=run_one, args=(name, url))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def fetch_cloudwatch_utilization(cluster: str, service: str, region: str, start_epoch: float, end_epoch: float) -> dict:
    import boto3

    cw = boto3.client("cloudwatch", region_name=region)
    metrics = {}
    for metric_name in ("CPUUtilization", "MemoryUtilization"):
        resp = cw.get_metric_statistics(
            Namespace="AWS/ECS",
            MetricName=metric_name,
            Dimensions=[
                {"Name": "ClusterName", "Value": cluster},
                {"Name": "ServiceName", "Value": service},
            ],
            StartTime=start_epoch,
            EndTime=end_epoch,
            Period=60,
            Statistics=["Average", "Maximum"],
        )
        points = sorted(resp.get("Datapoints", []), key=lambda d: d["Timestamp"])
        metrics[metric_name] = [
            {"t": p["Timestamp"].isoformat(), "avg": p["Average"], "max": p["Maximum"]} for p in points
        ]
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", required=True, help="name=base_url, repeatable")
    ap.add_argument("--rps", type=float, default=50.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--cluster", default=None, help="ECS cluster name for CloudWatch CPU/RAM pull")
    ap.add_argument("--region", default="eu-central-1")
    ap.add_argument("--out", default=None, help="write JSON report to this path")
    args = ap.parse_args()

    targets = {}
    for t in args.target:
        name, url = t.split("=", 1)
        targets[name] = url

    print(f"Running {args.rps} rps/service for {args.duration}s against: {targets}")
    t0 = time.time()
    results = run_concurrent(targets, args.rps, args.duration)
    t1 = time.time()

    report = {name: r.summary() for name, r in results.items()}

    if args.cluster:
        for name in targets:
            service_name = f"unitree-rl-{name}"
            try:
                report[name]["cloudwatch"] = fetch_cloudwatch_utilization(
                    args.cluster, service_name, args.region, t0, t1
                )
            except Exception as e:  # noqa: BLE001
                report[name]["cloudwatch_error"] = str(e)

    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

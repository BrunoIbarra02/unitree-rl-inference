#!/usr/bin/env bash
# Full acceptance benchmark suite (task spec section 6):
#   1. G1 alone
#   2. Go2 alone
#   3. Both concurrently
#   4. Sustained 10-minute run at 50 rps/service
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G1_URL=$(cat "$ROOT_DIR/.g1_url")
GO2_URL=$(cat "$ROOT_DIR/.go2_url")
CLUSTER_NAME=$(cat "$ROOT_DIR/.cluster_name")

REGION="${AWS_REGION:-eu-central-1}"
RPS="${RPS:-50}"
RESULTS_DIR="$ROOT_DIR/benchmarks/results"
mkdir -p "$RESULTS_DIR"
STAMP=$(date +%Y%m%d%H%M%S)

source "$ROOT_DIR/.bench_venv/bin/activate" 2>/dev/null || source "$ROOT_DIR/.bench_venv/Scripts/activate" 2>/dev/null || true

echo "=== 1/4: G1 alone (60s @ ${RPS} rps) ==="
python "$ROOT_DIR/benchmarks/load_test.py" --target "g1=$G1_URL" \
  --rps "$RPS" --duration 60 --cluster "$CLUSTER_NAME" --region "$REGION" \
  --out "$RESULTS_DIR/g1_alone_$STAMP.json"

echo "=== 2/4: Go2 alone (60s @ ${RPS} rps) ==="
python "$ROOT_DIR/benchmarks/load_test.py" --target "go2=$GO2_URL" \
  --rps "$RPS" --duration 60 --cluster "$CLUSTER_NAME" --region "$REGION" \
  --out "$RESULTS_DIR/go2_alone_$STAMP.json"

echo "=== 3/4: G1 + Go2 concurrently (60s @ ${RPS} rps each) ==="
python "$ROOT_DIR/benchmarks/load_test.py" --target "g1=$G1_URL" --target "go2=$GO2_URL" \
  --rps "$RPS" --duration 60 --cluster "$CLUSTER_NAME" --region "$REGION" \
  --out "$RESULTS_DIR/concurrent_$STAMP.json"

echo "=== 4/4: Sustained 10-minute run, both services concurrently @ ${RPS} rps each ==="
python "$ROOT_DIR/benchmarks/load_test.py" --target "g1=$G1_URL" --target "go2=$GO2_URL" \
  --rps "$RPS" --duration 600 --cluster "$CLUSTER_NAME" --region "$REGION" \
  --out "$RESULTS_DIR/sustained_10min_$STAMP.json"

echo "Results written to $RESULTS_DIR (stamp=$STAMP)"

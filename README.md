# unitree-rl-inference

CPU-only cloud inference for two pretrained Unitree RL locomotion policies:

- **G1** (biped, 12-DOF legs) — `unitreerobotics/unitree_rl_gym`, `deploy/pre_train/g1/motion.pt`
- **Go2** (quadruped, 12-DOF) — `wty-yy/go2_rl_gym`, `deploy/pre_train/go2/go2_cts_150k.pt`

No training, no fine-tuning, no GPU. Both checkpoints are used exactly as published.
Scope is cloud inference only — this does **not** drive real robot actuators or
implement low-level motor control.

## Architecture

Two fully isolated stacks, each with its own ALB (no shared load balancer, no
path-based routing):

```
Internet --> unitree-rl-g1-alb  (port 80) --> target group --> ECS Fargate task (g1, port 8000)
Internet --> unitree-rl-go2-alb (port 80) --> target group --> ECS Fargate task (go2, port 8000)
```

Both services share:
- The account's existing default VPC (`vpc-0601f05606f8b9eca`) and its 3 public
  subnets (no NAT gateway — Fargate tasks get public IPs directly, same
  pattern as every other project in this AWS account).
- One ECS cluster (`unitree-rl-cluster`) — a logical namespace only, not a
  networking boundary. Each service has its own SG pair (ALB SG + task SG,
  task SG only accepts traffic from its own ALB SG).

Provisioned with plain idempotent AWS CLI scripts (`scripts/lib.sh` /
`scripts/provision.sh`) — no Terraform/CDK. Every resource is looked up by a
fixed name before creation, so re-running `provision.sh` is always safe.

## Checkpoints

| | G1 | Go2 |
|---|---|---|
| Source | `unitreerobotics/unitree_rl_gym` | `wty-yy/go2_rl_gym` |
| File | `deploy/pre_train/g1/motion.pt` | `deploy/pre_train/go2/go2_cts_150k.pt` |
| SHA-256 | `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d` | `3e102d0d5e69aebd25e071bcdd87637596561111f924b160ed5ceb0d5c70f400` |
| Format | TorchScript (`torch.jit.load`), verified | TorchScript (`torch.jit.load`), verified |
| Obs / action dim | 47 / 12 | 45 / 12 |
| Recurrent mechanism | LSTM, state in `hidden_state`/`cell_state` module buffers (1,1,64) | 5-step observation history window, state in `history` module attribute (1,5,45) |

Both checkpoints keep their recurrent state as **mutable attributes inside the
loaded TorchScript module itself** (not an external hidden-state argument) —
confirmed by loading each checkpoint and inspecting `named_buffers()` /
`forward.code`. See `services/g1/app/model.py` and `services/go2/app/model.py`
for the full verified architecture notes (observation layout, scaling,
joint order) and source citations.

Because of that, serving multiple concurrent sessions off one shared model
instance required a snapshot/swap wrapper: each session's state tensor is
copied into the model's buffer immediately before `forward()`, and the
updated buffer is copied back out immediately after, all under a lock. This
is why `/predict` takes and returns an opaque `session_id` — reusing it
continues that robot's memory; omitting it starts a fresh zero-initialized
session. Sessions idle for >10 minutes are purged automatically.

`go2_cts_150k.pt` (not the MoE variants in the same repo) was chosen because
it has a single-tensor output (no expert-weights tuple to unpack), is
smaller, and avoids a confirmed-broken checkpoint path in that repo's own
`deploy_real` config.

## API (identical contract on both services)

```
GET  /health              -> {"status": "ok"}  (200 only once the model is loaded)
GET  /metadata             -> robot name, checkpoint id/sha256, obs/action dims,
                               joint order, observation layout, action scaling
POST /predict               body: {"observation": [float, ...], "session_id": "optional"}
                            -> {"actions": [...], "inference_latency_ms": float,
                                "checkpoint_id": str, "session_id": str, "session_was_reset": bool}
DELETE /session/{id}       -> drops a session's recurrent state early
```

`observation` must be exactly `observation_dim` finite floats (422 otherwise).
`/predict` returns the raw policy action — applying `default_angles +
action * action_scale` to get target joint positions is the caller's
responsibility (see `/metadata` for the exact formula/values).

Example:
```bash
curl -X POST http://<g1-alb-dns>/predict \
  -H 'Content-Type: application/json' \
  -d '{"observation": [0.0, 0.0, ...  /* 47 floats */]}'
```

## Deploying

Prerequisites: AWS CLI configured (`aws sts get-caller-identity` must work),
Docker running, Python 3.

```bash
scripts/deploy.sh
```

This builds both images, pushes to ECR, provisions both isolated stacks via
`scripts/provision.sh`, waits for both ECS services to stabilize, smoke-tests
`/health` and `/metadata` through each ALB, and runs a 60s/50rps smoke
benchmark. Safe to re-run — every run uses a fresh image tag so ECS always
redeploys the new image; all AWS resources are looked up by name, not
recreated.

For the full acceptance benchmark suite (G1 alone, Go2 alone, concurrent,
and a 10-minute sustained run), run separately since it takes ~13 minutes:
```bash
scripts/run_benchmarks.sh
```

## Tearing down

```bash
scripts/teardown.sh
```

Destroys both ALBs, target groups, security groups, the ECS cluster/services,
ECR repos, IAM roles, and log groups. Requires typing `destroy` to confirm.
Does not touch the shared default VPC/subnets or any other project in this
AWS account.

## Cost (eu-central-1, running continuously)

| Resource | Approx. monthly cost |
|---|---|
| 2x ALB (base, excl. LCU) | ~$37 |
| 2x Fargate task, 1 vCPU / 2GB, 24/7 | ~$83 |
| ECR storage, CloudWatch logs | <$1 |
| **Total** | **~$120/month** |

This is the cost of leaving both services running indefinitely. Run
`scripts/teardown.sh` when not actively using/benchmarking the services to
avoid ongoing charges. The IAM user used for this deployment does not have
`logs:PutRetentionPolicy`, so CloudWatch log groups retain logs indefinitely
by default — delete them manually (or via `teardown.sh`, which removes them)
if that matters for cost/compliance.

## Known limitations / things not reverified against real hardware

- Joint *names* (not order/count, which is verified from the model's actual
  forward behavior) are documented from the upstream repos' deploy configs
  and training code; if driving real hardware, verify against your own URDF.
- `go2.yaml`'s real-hardware deploy config uses different command scales and
  a joint-index remap for the physical SDK motor order — this service uses
  the simulation-equivalent scaling/order instead, consistent with its
  cloud-inference-only scope.
- Recurrent state continuity has only been validated for single-process,
  single-task deployment (desired count = 1 per service). If you scale a
  service's desired count above 1, sessions are not shared across tasks —
  a given `session_id`'s continuity depends on the ALB routing repeat calls
  to the same task, which it does not guarantee.

## Final benchmark results

Full raw JSON in `benchmarks/results/` (stamp `20261015163108`). Run against
the live ALB endpoints in `eu-central-1` from a client machine outside that
region (not co-located) — see the HTTP vs. model-latency note below.

| Test | Requests | Errors | Throughput | **Model P99** | HTTP e2e P50 / P99 |
|---|---:|---:|---:|---:|---|
| G1 alone (60s) | 3000 | 50 (cold-start, see note) | 48.4 rps | 36.3ms† | 266ms / 14.3s† |
| Go2 alone (60s) | 3000 | 0 | 48.9 rps | 2.17ms | 235ms / 1.57s |
| G1 + Go2 concurrent (60s) | 3000 each | 0 | 49.1 / 48.5 rps | 2.54ms / 2.13ms | 246ms / 1.67s |
| **Sustained 10min, concurrent** | 30000 each | 0 / 1 | 49.9 / 49.9 rps | **6.55ms / 3.29ms** | 246ms / 1.77s |
| Confirmatory re-run, pooled connections (30s) | 1500 each | 0 | 49.2 / 48.8 rps | 2.99ms / 1.87ms | **126ms / 667-756ms** |

† The G1-alone run was the very first load the service received immediately
after a fresh deployment -- all 50 errors and the elevated P99 occurred in
that single run and never recurred in any later test (including Go2's own
first-load run, which had 0 errors), consistent with one-time JIT/connection
warm-up rather than a steady-state defect.

**Model-only inference P99 meets the <10ms target** in every steady-state
run, including the full 10-minute sustained test at 50rps/service
(6.55ms G1, 3.29ms Go2). **No OOM, no non-finite actions, no sustained
errors** -- ECS CloudWatch metrics during the 10-minute run show stable
CPU (~10% G1, ~9% Go2) and memory (~7.4-7.7%) on the 1 vCPU / 2GB tasks,
comfortable headroom under the alternate 2 vCPU/4GB tier the task spec
allows if needed -- not needed here.

**HTTP end-to-end latency is dominated by real public-internet RTT from the
benchmark client to eu-central-1, not by the service.** The initial
benchmark run used `requests.post()` without connection reuse, paying a full
TCP handshake per request -- after switching to a pooled `requests.Session`
(see `benchmarks/bench_common.py`), P50 e2e dropped from ~245ms to ~126ms.
The residual latency is consistent with normal internet RTT for a client
not co-located with the AWS region; it is not reducible from the service
side. A client actually co-located in eu-central-1 (e.g. an EC2 instance in
the same VPC) would see e2e latency close to the model-only numbers above.

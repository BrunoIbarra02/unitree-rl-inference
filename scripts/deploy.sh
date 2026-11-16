#!/usr/bin/env bash
# Reproducible deploy for the unitree-rl-inference G1 + Go2 services.
# Each service is a fully isolated stack (its own ALB, SGs, target group,
# ECS service) -- see scripts/lib.sh / provision.sh. No Terraform; plain
# idempotent AWS CLI calls. Safe to re-run: every run uses a fresh image
# tag so ECS always deploys the new image.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGION="${AWS_REGION:-eu-central-1}"
IMAGE_TAG="$(date +%Y%m%d%H%M%S)-$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)"

log() { echo -e "\n\033[1;36m[deploy]\033[0m $*"; }
die() { echo -e "\033[1;31m[deploy][ERROR]\033[0m $*" >&2; exit 1; }

log "Step 1/9: Validating AWS credentials and region"
CALLER_ARN=$(aws sts get-caller-identity --query Arn --output text --region "$REGION") \
  || die "AWS credentials invalid or not configured. Run 'aws configure' first."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
log "Authenticated as $CALLER_ARN in account $ACCOUNT_ID, region $REGION"

log "Step 2/9: Building Docker images (tag=$IMAGE_TAG)"
for svc in g1 go2; do
  docker build -t "unitree-rl-$svc:$IMAGE_TAG" -f "$ROOT_DIR/services/$svc/Dockerfile" "$ROOT_DIR/services/$svc" \
    || die "Docker build failed for $svc"
done

log "Step 3/9: Logging into ECR and pushing images"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin \
  "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com" || die "ECR login failed"

for svc in g1 go2; do
  REPO_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/unitree-rl-$svc"
  aws ecr describe-repositories --region "$REGION" --repository-names "unitree-rl-$svc" >/dev/null 2>&1 \
    || aws ecr create-repository --region "$REGION" --repository-name "unitree-rl-$svc" \
         --image-scanning-configuration scanOnPush=true >/dev/null
  docker tag "unitree-rl-$svc:$IMAGE_TAG" "$REPO_URI:$IMAGE_TAG"
  docker push "$REPO_URI:$IMAGE_TAG" || die "Docker push failed for $svc"
done

log "Step 4/9 & 5/9: Provisioning AWS infra (2 isolated ALBs/SGs/target groups, ECS cluster/services)"
bash "$ROOT_DIR/scripts/provision.sh" "$IMAGE_TAG" || die "Provisioning failed"

G1_URL=$(cat "$ROOT_DIR/.g1_url")
GO2_URL=$(cat "$ROOT_DIR/.go2_url")
CLUSTER_NAME=$(cat "$ROOT_DIR/.cluster_name")

log "Step 6/9: ALB listener configuration applied as part of provisioning above"
log "G1 ALB:  $G1_URL"
log "Go2 ALB: $GO2_URL"

log "Step 7/9: ECS services already confirmed stable by provision.sh (ecs wait services-stable)"

log "Step 8/9: Smoke-testing public endpoints"
for name in g1 go2; do
  URL_VAR="${name^^}_URL"
  URL="${!URL_VAR}"
  log "GET $URL/health"
  curl -fsS "$URL/health" || die "$name /health check failed via ALB"
  echo
  log "GET $URL/metadata"
  curl -fsS "$URL/metadata" | tee "/tmp/${name}_metadata.json" || die "$name /metadata failed via ALB"
  echo
done

log "Step 9/9: Running smoke-level inference benchmark (60s @ 50rps/service)"
cd "$ROOT_DIR"
python3 -m venv .bench_venv 2>/dev/null || true
source .bench_venv/bin/activate 2>/dev/null || source .bench_venv/Scripts/activate
pip install -q -r benchmarks/requirements.txt
mkdir -p benchmarks/results
python benchmarks/load_test.py \
  --target "g1=$G1_URL" --target "go2=$GO2_URL" \
  --rps 50 --duration 60 --cluster "$CLUSTER_NAME" --region "$REGION" \
  --out "benchmarks/results/smoke_$IMAGE_TAG.json"

log "Deploy complete."
echo "G1  URL: $G1_URL"
echo "Go2 URL: $GO2_URL"
echo "Image tag: $IMAGE_TAG"
echo "Run 'scripts/run_benchmarks.sh' for the full acceptance benchmark suite (incl. 10-min sustained test)."

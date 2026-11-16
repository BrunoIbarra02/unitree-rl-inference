#!/usr/bin/env bash
# Shared idempotent AWS CLI helpers for provision.sh / teardown.sh.
# No Terraform, no state file -- every resource is looked up by its fixed
# name before creating, so re-running provision.sh is always safe.
set -euo pipefail

# Git Bash/MSYS rewrites any argument starting with "/" (e.g. CloudWatch log
# group names like "/ecs/unitree-rl-g1") into a Windows path before it
# reaches aws.exe, corrupting it. Disable that globally for this script.
export MSYS_NO_PATHCONV=1

REGION="${AWS_REGION:-eu-central-1}"
PROJECT="unitree-rl"
VPC_ID="vpc-0601f05606f8b9eca"
SUBNET_IDS=(subnet-0e912e123cfa748e6 subnet-0d60f3805c6e3b93b subnet-0cb4eeb17db1d3d80)
CONTAINER_PORT=8000
TASK_CPU=1024
TASK_MEMORY=2048

log() { echo -e "\033[1;36m[infra]\033[0m $*" >&2; } # stderr: keeps log lines out of $(...) command-substitution captures

account_id() { aws sts get-caller-identity --query Account --output text --region "$REGION"; }

ecr_repo_uri() {
  local svc="$1"
  echo "$(account_id).dkr.ecr.$REGION.amazonaws.com/$PROJECT-$svc"
}

ensure_ecr_repo() {
  local svc="$1"
  local name="$PROJECT-$svc"
  if aws ecr describe-repositories --region "$REGION" --repository-names "$name" >/dev/null 2>&1; then
    log "ECR repo $name exists"
  else
    log "Creating ECR repo $name"
    aws ecr create-repository --region "$REGION" --repository-name "$name" \
      --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE >/dev/null
  fi
}

ensure_log_group() {
  local svc="$1"
  local name="/ecs/$PROJECT-$svc"
  if aws logs describe-log-groups --region "$REGION" --log-group-name-prefix "$name" \
      --query "logGroups[?logGroupName=='$name']" --output text | grep -q .; then
    log "Log group $name exists"
  else
    log "Creating log group $name"
    aws logs create-log-group --region "$REGION" --log-group-name "$name"
    aws logs put-retention-policy --region "$REGION" --log-group-name "$name" --retention-in-days 14 \
      || log "  (no logs:PutRetentionPolicy permission -- log group created, but will retain logs indefinitely)"
  fi
}

ensure_iam_role() {
  local role_name="$1"
  local managed_policy_arn="${2:-}"
  if aws iam get-role --role-name "$role_name" >/dev/null 2>&1; then
    log "IAM role $role_name exists"
  else
    log "Creating IAM role $role_name"
    aws iam create-role --role-name "$role_name" --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{"Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }' >/dev/null
    sleep 5 # IAM eventual consistency before policy attach / first use
  fi
  if [[ -n "$managed_policy_arn" ]]; then
    aws iam attach-role-policy --role-name "$role_name" --policy-arn "$managed_policy_arn" >/dev/null 2>&1 || true
  fi
}

ensure_security_group() {
  # ensure_security_group <name> <description>
  local name="$1" desc="$2"
  local sg_id
  sg_id=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=$name" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
  if [[ "$sg_id" != "None" && -n "$sg_id" ]]; then
    echo "$sg_id"
    return
  fi
  sg_id=$(aws ec2 create-security-group --region "$REGION" --group-name "$name" \
    --description "$desc" --vpc-id "$VPC_ID" --query 'GroupId' --output text)
  aws ec2 authorize-security-group-egress --region "$REGION" --group-id "$sg_id" \
    --ip-permissions 'IpProtocol=-1,IpRanges=[{CidrIp=0.0.0.0/0}]' >/dev/null 2>&1 || true
  echo "$sg_id"
}

sg_authorize_ingress_cidr() {
  local sg_id="$1" port="$2" cidr="$3"
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$sg_id" \
    --ip-permissions "IpProtocol=tcp,FromPort=$port,ToPort=$port,IpRanges=[{CidrIp=$cidr}]" >/dev/null 2>&1 || true
}

sg_authorize_ingress_sg() {
  local sg_id="$1" port="$2" source_sg="$3"
  aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$sg_id" \
    --ip-permissions "IpProtocol=tcp,FromPort=$port,ToPort=$port,UserIdGroupPairs=[{GroupId=$source_sg}]" >/dev/null 2>&1 || true
}

ensure_alb() {
  # ensure_alb <name> <sg_id> -- echoes "<arn> <dns_name>"
  local name="$1" sg_id="$2"
  local arn dns
  arn=$(aws elbv2 describe-load-balancers --region "$REGION" --names "$name" \
    --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null) || arn="None"
  if [[ "$arn" != "None" && -n "$arn" ]]; then
    dns=$(aws elbv2 describe-load-balancers --region "$REGION" --load-balancer-arns "$arn" \
      --query 'LoadBalancers[0].DNSName' --output text)
    echo "$arn $dns"
    return
  fi
  log "Creating ALB $name"
  arn=$(aws elbv2 create-load-balancer --region "$REGION" --name "$name" \
    --subnets "${SUBNET_IDS[@]}" --security-groups "$sg_id" --scheme internet-facing --type application \
    --query 'LoadBalancers[0].LoadBalancerArn' --output text)
  aws elbv2 wait load-balancer-available --region "$REGION" --load-balancer-arns "$arn"
  dns=$(aws elbv2 describe-load-balancers --region "$REGION" --load-balancer-arns "$arn" \
    --query 'LoadBalancers[0].DNSName' --output text)
  echo "$arn $dns"
}

ensure_target_group() {
  local name="$1"
  local arn
  arn=$(aws elbv2 describe-target-groups --region "$REGION" --names "$name" \
    --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null) || arn="None"
  if [[ "$arn" != "None" && -n "$arn" ]]; then
    echo "$arn"
    return
  fi
  log "Creating target group $name"
  aws elbv2 create-target-group --region "$REGION" --name "$name" \
    --protocol HTTP --port "$CONTAINER_PORT" --vpc-id "$VPC_ID" --target-type ip \
    --health-check-path /health --health-check-protocol HTTP \
    --healthy-threshold-count 2 --unhealthy-threshold-count 3 \
    --health-check-interval-seconds 15 --health-check-timeout-seconds 5 \
    --query 'TargetGroups[0].TargetGroupArn' --output text
}

ensure_listener() {
  # ensure_listener <alb_arn> <tg_arn> -- default forward, no path rules (dedicated ALB per service)
  local alb_arn="$1" tg_arn="$2"
  local listener_arn
  listener_arn=$(aws elbv2 describe-listeners --region "$REGION" --load-balancer-arn "$alb_arn" \
    --query 'Listeners[0].ListenerArn' --output text 2>/dev/null) || listener_arn="None"
  if [[ "$listener_arn" != "None" && -n "$listener_arn" ]]; then
    echo "$listener_arn"
    return
  fi
  log "Creating listener on $alb_arn"
  aws elbv2 create-listener --region "$REGION" --load-balancer-arn "$alb_arn" \
    --protocol HTTP --port 80 --default-actions "Type=forward,TargetGroupArn=$tg_arn" \
    --query 'Listeners[0].ListenerArn' --output text
}

ensure_cluster() {
  local name="$1"
  local status
  status=$(aws ecs describe-clusters --region "$REGION" --clusters "$name" \
    --query 'clusters[0].status' --output text 2>/dev/null) || status="MISSING"
  if [[ "$status" == "ACTIVE" ]]; then
    log "ECS cluster $name exists"
  else
    log "Creating ECS cluster $name"
    aws ecs create-cluster --region "$REGION" --cluster-name "$name" >/dev/null
  fi
}

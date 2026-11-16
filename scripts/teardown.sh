#!/usr/bin/env bash
# Reproducible teardown: deletes every resource provision.sh created for
# BOTH services (ECS services/cluster, ALBs, listeners, target groups, SGs,
# ECR repos, IAM roles, log groups). Does not touch the shared default
# VPC/subnets or any other project's infra in this account.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

echo "This will destroy ALL unitree-rl-inference AWS resources (2 ALBs, ECS cluster+services, ECR repos, SGs, IAM roles)."
read -p "Type 'destroy' to confirm: " CONFIRM
if [[ "$CONFIRM" != "destroy" ]]; then
  echo "Aborted."
  exit 1
fi

CLUSTER_NAME="$PROJECT-cluster"

for svc in g1 go2; do
  log "=== Tearing down $svc ==="

  if aws ecs describe-services --region "$REGION" --cluster "$CLUSTER_NAME" --services "$PROJECT-$svc" \
      --query "services[?status=='ACTIVE']" --output text 2>/dev/null | grep -q .; then
    log "Scaling down and deleting ECS service $PROJECT-$svc"
    aws ecs update-service --region "$REGION" --cluster "$CLUSTER_NAME" --service "$PROJECT-$svc" --desired-count 0 >/dev/null
    aws ecs wait services-stable --region "$REGION" --cluster "$CLUSTER_NAME" --services "$PROJECT-$svc" || true
    aws ecs delete-service --region "$REGION" --cluster "$CLUSTER_NAME" --service "$PROJECT-$svc" >/dev/null
  fi

  ALB_ARN=$(aws elbv2 describe-load-balancers --region "$REGION" --names "$PROJECT-$svc-alb" \
    --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null) || ALB_ARN="None"
  if [[ "$ALB_ARN" != "None" && -n "$ALB_ARN" ]]; then
    LISTENER_ARN=$(aws elbv2 describe-listeners --region "$REGION" --load-balancer-arn "$ALB_ARN" \
      --query 'Listeners[0].ListenerArn' --output text 2>/dev/null) || true
    [[ -n "${LISTENER_ARN:-}" && "$LISTENER_ARN" != "None" ]] && \
      aws elbv2 delete-listener --region "$REGION" --listener-arn "$LISTENER_ARN" >/dev/null 2>&1 || true
    log "Deleting ALB $PROJECT-$svc-alb"
    aws elbv2 delete-load-balancer --region "$REGION" --load-balancer-arn "$ALB_ARN" >/dev/null
    aws elbv2 wait load-balancers-deleted --region "$REGION" --load-balancer-arns "$ALB_ARN" 2>/dev/null || sleep 20
  fi

  TG_ARN=$(aws elbv2 describe-target-groups --region "$REGION" --names "$PROJECT-$svc-tg" \
    --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null) || TG_ARN="None"
  if [[ "$TG_ARN" != "None" && -n "$TG_ARN" ]]; then
    log "Deleting target group $PROJECT-$svc-tg"
    aws elbv2 delete-target-group --region "$REGION" --target-group-arn "$TG_ARN" >/dev/null 2>&1 || true
  fi

  for sg_name in "$PROJECT-$svc-tasks-sg" "$PROJECT-$svc-alb-sg"; do
    SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
      --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=$sg_name" \
      --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null) || SG_ID="None"
    if [[ "$SG_ID" != "None" && -n "$SG_ID" ]]; then
      log "Deleting security group $sg_name"
      aws ec2 delete-security-group --region "$REGION" --group-id "$SG_ID" >/dev/null 2>&1 || \
        log "  (could not delete $sg_name yet, may still be attached -- retry teardown in a minute)"
    fi
  done

  aws ecr delete-repository --region "$REGION" --repository-name "$PROJECT-$svc" --force >/dev/null 2>&1 || true
  aws logs delete-log-group --region "$REGION" --log-group-name "/ecs/$PROJECT-$svc" >/dev/null 2>&1 || true
done

log "Deleting ECS cluster $CLUSTER_NAME"
aws ecs delete-cluster --region "$REGION" --cluster "$CLUSTER_NAME" >/dev/null 2>&1 || true

for role in "$PROJECT-ecs-execution-role" "$PROJECT-ecs-task-role"; do
  POLICIES=$(aws iam list-attached-role-policies --role-name "$role" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null) || POLICIES=""
  for p in $POLICIES; do aws iam detach-role-policy --role-name "$role" --policy-arn "$p" >/dev/null 2>&1 || true; done
  aws iam delete-role --role-name "$role" >/dev/null 2>&1 || true
done

rm -f "$SCRIPT_DIR/../.g1_url" "$SCRIPT_DIR/../.go2_url" "$SCRIPT_DIR/../.cluster_name"
log "Teardown complete."

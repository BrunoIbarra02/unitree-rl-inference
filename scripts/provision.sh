#!/usr/bin/env bash
# Idempotent AWS CLI provisioning for unitree-rl-inference. No Terraform.
# Each service (g1, go2) gets a fully isolated stack: its own ALB, security
# groups, target group, task definition, and ECS service. They only share
# the pre-existing default VPC/subnets and a single ECS cluster (a logical
# namespace -- not a networking boundary).
#
# Usage: scripts/provision.sh <image_tag>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib.sh"

IMAGE_TAG="${1:?usage: provision.sh <image_tag>}"
CLUSTER_NAME="$PROJECT-cluster"

log "Account: $(account_id)  Region: $REGION"

ensure_cluster "$CLUSTER_NAME"
ensure_iam_role "$PROJECT-ecs-execution-role" "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
ensure_iam_role "$PROJECT-ecs-task-role"
EXEC_ROLE_ARN="arn:aws:iam::$(account_id):role/$PROJECT-ecs-execution-role"
TASK_ROLE_ARN="arn:aws:iam::$(account_id):role/$PROJECT-ecs-task-role"

declare -A SERVICE_URL

for svc in g1 go2; do
  log "=== Provisioning $svc ==="
  ensure_ecr_repo "$svc"
  ensure_log_group "$svc"

  ALB_SG=$(ensure_security_group "$PROJECT-$svc-alb-sg" "Public ingress for $PROJECT $svc ALB")
  sg_authorize_ingress_cidr "$ALB_SG" 80 "0.0.0.0/0"

  TASK_SG=$(ensure_security_group "$PROJECT-$svc-tasks-sg" "ECS task ingress restricted to $PROJECT $svc ALB only")
  sg_authorize_ingress_sg "$TASK_SG" "$CONTAINER_PORT" "$ALB_SG"

  read -r ALB_ARN ALB_DNS <<< "$(ensure_alb "$PROJECT-$svc-alb" "$ALB_SG")"
  TG_ARN=$(ensure_target_group "$PROJECT-$svc-tg")
  ensure_listener "$ALB_ARN" "$TG_ARN" >/dev/null

  IMAGE_URI="$(ecr_repo_uri "$svc"):$IMAGE_TAG"
  CONTAINER_DEF=$(cat <<EOF
[{
  "name": "$svc",
  "image": "$IMAGE_URI",
  "essential": true,
  "portMappings": [{"containerPort": $CONTAINER_PORT, "protocol": "tcp"}],
  "environment": [
    {"name": "ROBOT_NAME", "value": "$svc"},
    {"name": "OMP_NUM_THREADS", "value": "1"}
  ],
  "logConfiguration": {
    "logDriver": "awslogs",
    "options": {
      "awslogs-group": "/ecs/$PROJECT-$svc",
      "awslogs-region": "$REGION",
      "awslogs-stream-prefix": "$svc"
    }
  }
}]
EOF
)
  log "Registering task definition $PROJECT-$svc (image tag $IMAGE_TAG)"
  TASK_DEF_ARN=$(aws ecs register-task-definition --region "$REGION" \
    --family "$PROJECT-$svc" \
    --requires-compatibilities FARGATE --network-mode awsvpc \
    --cpu "$TASK_CPU" --memory "$TASK_MEMORY" \
    --execution-role-arn "$EXEC_ROLE_ARN" --task-role-arn "$TASK_ROLE_ARN" \
    --container-definitions "$CONTAINER_DEF" \
    --query 'taskDefinition.taskDefinitionArn' --output text)

  SUBNETS_JSON=$(printf '"%s",' "${SUBNET_IDS[@]}" | sed 's/,$//')
  SERVICE_EXISTS=$(aws ecs describe-services --region "$REGION" --cluster "$CLUSTER_NAME" --services "$PROJECT-$svc" \
    --query "services[?status=='ACTIVE'].serviceName" --output text 2>/dev/null)

  if [[ -n "$SERVICE_EXISTS" ]]; then
    log "Updating ECS service $PROJECT-$svc to new task definition"
    aws ecs update-service --region "$REGION" --cluster "$CLUSTER_NAME" --service "$PROJECT-$svc" \
      --task-definition "$TASK_DEF_ARN" --force-new-deployment >/dev/null
  else
    log "Creating ECS service $PROJECT-$svc"
    aws ecs create-service --region "$REGION" --cluster "$CLUSTER_NAME" --service-name "$PROJECT-$svc" \
      --task-definition "$TASK_DEF_ARN" --desired-count 1 --launch-type FARGATE \
      --network-configuration "{\"awsvpcConfiguration\":{\"subnets\":[$SUBNETS_JSON],\"securityGroups\":[\"$TASK_SG\"],\"assignPublicIp\":\"ENABLED\"}}" \
      --load-balancers "targetGroupArn=$TG_ARN,containerName=$svc,containerPort=$CONTAINER_PORT" \
      >/dev/null
  fi

  SERVICE_URL[$svc]="http://$ALB_DNS"
  log "$svc ALB DNS: $ALB_DNS"
done

log "Waiting for both services to reach steady state..."
aws ecs wait services-stable --region "$REGION" --cluster "$CLUSTER_NAME" --services "$PROJECT-g1" "$PROJECT-go2"

echo
echo "=== Provisioning complete ==="
echo "G1  URL: ${SERVICE_URL[g1]}"
echo "Go2 URL: ${SERVICE_URL[go2]}"
echo "${SERVICE_URL[g1]}" > "$SCRIPT_DIR/../.g1_url"
echo "${SERVICE_URL[go2]}" > "$SCRIPT_DIR/../.go2_url"
echo "$CLUSTER_NAME" > "$SCRIPT_DIR/../.cluster_name"

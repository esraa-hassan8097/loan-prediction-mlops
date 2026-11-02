#!/usr/bin/env bash
set -euo pipefail

IMAGE="$1"

echo "Candidate image: $IMAGE"

ACTIVE_COLOR=$(kubectl get service loan-app-service \
  -o jsonpath='{.spec.selector.version}' 2>/dev/null || true)

if [ "$ACTIVE_COLOR" = "blue" ]; then
    TARGET_COLOR="green"
else
    TARGET_COLOR="blue"
fi

echo "Currently active: ${ACTIVE_COLOR:-none}"
echo "Target color: $TARGET_COLOR"

kubectl set image \
  deployment/loan-app-$TARGET_COLOR \
  loan-app="$IMAGE"

kubectl scale \
  deployment/loan-app-$TARGET_COLOR \
  --replicas=2

kubectl rollout status \
  deployment/loan-app-$TARGET_COLOR \
  --timeout=120s

echo "$TARGET_COLOR is healthy."

kubectl patch service loan-app-service \
  -p "{\"spec\":{\"selector\":{\"app\":\"loan-app\",\"version\":\"$TARGET_COLOR\"}}}"

echo "Traffic switched to $TARGET_COLOR."

kubectl get service loan-app-service \
  -o jsonpath='{.spec.selector}{"\n"}'

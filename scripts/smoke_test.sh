#!/bin/bash
set -e

HOST="${1:-http://localhost:8000}"

echo "Running smoke tests against $HOST"

echo "1. Checking /health endpoint..."
health_status=$(curl -s -o /dev/null -w "%{http_code}" "$HOST/health")
if [ "$health_status" != "200" ]; then
  echo "FAILED: /health returned $health_status"
  exit 1
fi
echo "PASSED: /health returned 200"

echo "2. Checking /predict endpoint..."
predict_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$HOST/predict" -F "file=@test_image.jpg")
if [ "$predict_status" != "200" ]; then
  echo "FAILED: /predict returned $predict_status"
  exit 1
fi
echo "PASSED: /predict returned 200"

echo "All smoke tests passed."
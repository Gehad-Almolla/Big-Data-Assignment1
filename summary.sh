#!/bin/bash
set -e

CONTAINER_NAME="${1:-customer-analytics-container}"

mkdir -p customer-analytics/results
docker cp "${CONTAINER_NAME}:/app/pipeline/results/." customer-analytics/results/
docker stop "${CONTAINER_NAME}"
docker rm "${CONTAINER_NAME}"

echo "Results copied to customer-analytics/results/ and container removed."

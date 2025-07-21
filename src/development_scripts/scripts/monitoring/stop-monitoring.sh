#!/bin/bash
# Stop anomaly_detection Monitoring Stack

echo "🛑 Stopping anomaly_detection monitoring stack..."

# Stop Python services
pkill -f "anomaly_detection.infrastructure.monitoring"

# Stop Docker services
docker-compose -f docker-compose.monitoring.yml down

echo "✅ Monitoring stack stopped!"

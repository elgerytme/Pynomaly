#!/bin/bash
# Stop Pynomaly Monitoring Stack

echo "🛑 Stopping Pynomaly monitoring stack..."

# Stop Python services
pkill -f "pynomaly.infrastructure.monitoring"

# Stop Docker services
docker-compose -f docker-compose.monitoring.yml down

echo "✅ Monitoring stack stopped!"

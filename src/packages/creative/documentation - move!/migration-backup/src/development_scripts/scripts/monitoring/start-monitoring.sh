#!/bin/bash
# Start Pynomaly Monitoring Stack

echo "🔄 Starting Pynomaly monitoring stack..."

# Start Docker services
docker-compose -f docker-compose.monitoring.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
echo "Prometheus: $(curl -s http://localhost:9090/-/healthy || echo 'NOT READY')"
echo "Grafana: $(curl -s http://localhost:3000/api/health || echo 'NOT READY')"
echo "Alertmanager: $(curl -s http://localhost:9093/-/healthy || echo 'NOT READY')"

# Start Python monitoring services
echo "🐍 Starting Python monitoring services..."
python3 -m pynomaly.infrastructure.monitoring.alerts &
python3 -m pynomaly.infrastructure.monitoring.dashboard &

echo "✅ Monitoring stack started!"
echo "📊 Access points:"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Alertmanager: http://localhost:9093"
echo "  - Dashboard: http://localhost:8080"

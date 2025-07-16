#!/bin/bash
# Check Pynomaly Monitoring Stack Status

echo "📊 Pynomaly Monitoring Stack Status"
echo "=================================="

# Check Docker services
echo "🐳 Docker Services:"
docker-compose -f docker-compose.monitoring.yml ps

echo ""
echo "🔗 Service Endpoints:"
echo "  - Grafana: http://localhost:3000"
echo "  - Prometheus: http://localhost:9090"
echo "  - Alertmanager: http://localhost:9093"
echo "  - Dashboard: http://localhost:8080"

echo ""
echo "🏥 Health Checks:"
echo "  - Prometheus: $(curl -s http://localhost:9090/-/healthy 2>/dev/null || echo 'DOWN')"
echo "  - Grafana: $(curl -s http://localhost:3000/api/health 2>/dev/null | grep -o '"database":"ok"' || echo 'DOWN')"
echo "  - Alertmanager: $(curl -s http://localhost:9093/-/healthy 2>/dev/null || echo 'DOWN')"
echo "  - Dashboard: $(curl -s http://localhost:8080/health 2>/dev/null | grep -o '"status":"healthy"' || echo 'DOWN')"

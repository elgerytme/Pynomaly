#!/bin/bash

# Production Environment - Docker Run Scripts
# Optimized production deployment with security and monitoring

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default configuration
ENV_FILE="${PROJECT_ROOT}/.env.prod"
NETWORK_NAME="anomaly_detection-prod"
CONTAINER_NAME="anomaly_detection-prod"
IMAGE_NAME="anomaly_detection:latest"
HOST_PORT=80
CONTAINER_PORT=8000
REPLICAS=1

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -p, --port PORT      Host port to bind (default: 80)"
    echo "  -r, --replicas NUM   Number of replicas (default: 1)"
    echo "  -i, --image IMAGE    Docker image name (default: anomaly_detection:latest)"
    echo "  --ssl                Enable SSL/TLS (requires certificates)"
    echo "  --monitoring         Include monitoring stack"
    echo "  --logging            Include centralized logging"
    echo "  --build              Build production image before running"
    echo "  --clean              Remove existing containers first"
    echo "  --stop               Stop all production containers"
    echo "  -h, --help           Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            HOST_PORT="$2"
            shift 2
            ;;
        -r|--replicas)
            REPLICAS="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --ssl)
            SSL_ENABLED=true
            shift
            ;;
        --monitoring)
            MONITORING=true
            shift
            ;;
        --logging)
            LOGGING=true
            shift
            ;;
        --build)
            BUILD_IMAGE=true
            shift
            ;;
        --clean)
            CLEAN_CONTAINERS=true
            shift
            ;;
        --stop)
            STOP_CONTAINERS=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Function to stop all production containers
stop_containers() {
    echo "Stopping all production containers..."
    for i in $(seq 1 $REPLICAS); do
        docker rm -f "${CONTAINER_NAME}-${i}" 2>/dev/null || true
    done
    docker rm -f "anomaly_detection-nginx-prod" 2>/dev/null || true
    docker rm -f "anomaly_detection-prometheus-prod" 2>/dev/null || true
    docker rm -f "anomaly_detection-grafana-prod" 2>/dev/null || true
    docker rm -f "anomaly_detection-elasticsearch-prod" 2>/dev/null || true
    docker rm -f "anomaly_detection-logstash-prod" 2>/dev/null || true
    docker rm -f "anomaly_detection-kibana-prod" 2>/dev/null || true
    echo "All production containers stopped."
    exit 0
}

if [[ "$STOP_CONTAINERS" == "true" ]]; then
    stop_containers
fi

# Create network
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Clean up if requested
if [[ "$CLEAN_CONTAINERS" == "true" ]]; then
    echo "Cleaning up existing production containers..."
    stop_containers
fi

# Build production image if requested
if [[ "$BUILD_IMAGE" == "true" ]]; then
    echo "Building production image..."
    docker build -t "$IMAGE_NAME" -f "$PROJECT_ROOT/Dockerfile.hardened" "$PROJECT_ROOT"
fi

# Start monitoring stack if requested
if [[ "$MONITORING" == "true" ]]; then
    echo "Starting monitoring stack..."

    # Start Prometheus
    docker run -d \
        --name "anomaly_detection-prometheus-prod" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 9090:9090 \
        -v "${PROJECT_ROOT}/docker/monitoring/prometheus-prod.yml:/etc/prometheus/prometheus.yml" \
        -v prometheus-prod-data:/prometheus \
        prom/prometheus:latest \
        --config.file=/etc/prometheus/prometheus.yml \
        --storage.tsdb.path=/prometheus \
        --web.console.libraries=/usr/share/prometheus/console_libraries \
        --web.console.templates=/usr/share/prometheus/consoles \
        --storage.tsdb.retention.time=30d \
        --web.enable-lifecycle

    # Start Grafana
    docker run -d \
        --name "anomaly_detection-grafana-prod" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 3000:3000 \
        -e GF_SECURITY_ADMIN_PASSWORD_FILE=/run/secrets/grafana_password \
        -v grafana-prod-data:/var/lib/grafana \
        -v "${PROJECT_ROOT}/docker/monitoring/grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml" \
        grafana/grafana:latest
fi

# Start logging stack if requested
if [[ "$LOGGING" == "true" ]]; then
    echo "Starting logging stack..."

    # Start Elasticsearch
    docker run -d \
        --name "anomaly_detection-elasticsearch-prod" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 9200:9200 \
        -e "discovery.type=single-node" \
        -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
        -v elasticsearch-prod-data:/usr/share/elasticsearch/data \
        elasticsearch:8.11.0

    # Start Logstash
    docker run -d \
        --name "anomaly_detection-logstash-prod" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 5044:5044 \
        -v "${PROJECT_ROOT}/docker/logging/logstash.conf:/usr/share/logstash/pipeline/logstash.conf" \
        logstash:8.11.0

    # Start Kibana
    docker run -d \
        --name "anomaly_detection-kibana-prod" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        -p 5601:5601 \
        -e ELASTICSEARCH_HOSTS=http://anomaly_detection-elasticsearch-prod:9200 \
        kibana:8.11.0
fi

# Start Nginx reverse proxy
echo "Starting Nginx reverse proxy..."
NGINX_CONFIG="nginx-prod.conf"
if [[ "$SSL_ENABLED" == "true" ]]; then
    NGINX_CONFIG="nginx-prod-ssl.conf"
fi

docker run -d \
    --name "anomaly_detection-nginx-prod" \
    --network "$NETWORK_NAME" \
    --restart unless-stopped \
    -p "${HOST_PORT}:80" \
    $(if [[ "$SSL_ENABLED" == "true" ]]; then echo "-p 443:443"; fi) \
    -v "${PROJECT_ROOT}/docker/nginx/${NGINX_CONFIG}:/etc/nginx/nginx.conf" \
    $(if [[ "$SSL_ENABLED" == "true" ]]; then echo "-v ${PROJECT_ROOT}/certs:/etc/nginx/certs"; fi) \
    nginx:alpine

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Start application replicas
echo "Starting $REPLICAS application replicas..."
for i in $(seq 1 $REPLICAS); do
    APP_PORT=$((8000 + i - 1))
    echo "Starting replica $i on port $APP_PORT..."

    docker run -d \
        --name "${CONTAINER_NAME}-${i}" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        --env-file "$ENV_FILE" \
        -p "${APP_PORT}:8000" \
        -v "${PROJECT_ROOT}/storage:/app/storage" \
        -v "${PROJECT_ROOT}/logs:/app/logs" \
        -e PYTHONPATH=/app/src \
        -e ENVIRONMENT=production \
        -e DEBUG=false \
        -e LOG_LEVEL=INFO \
        -e REPLICA_ID="$i" \
        -e PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc \
        --security-opt no-new-privileges:true \
        --read-only \
        --tmpfs /tmp \
        --tmpfs /app/logs \
        --user 1000:1000 \
        --memory 2g \
        --cpus 1.0 \
        --health-cmd "curl -f http://localhost:8000/health || exit 1" \
        --health-interval 30s \
        --health-timeout 10s \
        --health-retries 3 \
        "$IMAGE_NAME" \
        poetry run gunicorn anomaly_detection.presentation.api:app \
        --bind 0.0.0.0:8000 \
        --workers 4 \
        --worker-class uvicorn.workers.UvicornWorker \
        --worker-connections 1000 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --timeout 30 \
        --keep-alive 2 \
        --access-logfile /app/logs/access.log \
        --error-logfile /app/logs/error.log \
        --log-level info
done

echo "Production deployment completed!"
echo "Application: http://localhost:${HOST_PORT}"
if [[ "$SSL_ENABLED" == "true" ]]; then
    echo "HTTPS: https://localhost:443"
fi
if [[ "$MONITORING" == "true" ]]; then
    echo "Monitoring: http://localhost:3000"
    echo "Metrics: http://localhost:9090"
fi
if [[ "$LOGGING" == "true" ]]; then
    echo "Logs: http://localhost:5601"
fi
echo "Replicas: $REPLICAS"
echo ""
echo "Use --stop to stop all containers"

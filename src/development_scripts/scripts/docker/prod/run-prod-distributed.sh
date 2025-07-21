#!/bin/bash

# Distributed Production Environment
# Multi-node deployment with load balancing and service discovery

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default configuration
ENV_FILE="${PROJECT_ROOT}/.env.prod"
NETWORK_NAME="anomaly_detection-distributed"
IMAGE_NAME="anomaly_detection:latest"
WORKERS=3
API_REPLICAS=2
STORAGE_TYPE="postgres"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --workers NUM        Number of worker nodes (default: 3)"
    echo "  --api-replicas NUM   Number of API replicas (default: 2)"
    echo "  --storage TYPE       Storage backend (postgres|mongodb|redis-cluster) (default: postgres)"
    echo "  --consul             Enable service discovery with Consul"
    echo "  --vault              Enable secrets management with Vault"
    echo "  --build              Build production image before running"
    echo "  --clean              Remove existing containers first"
    echo "  --stop               Stop all distributed containers"
    echo "  -h, --help           Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --api-replicas)
            API_REPLICAS="$2"
            shift 2
            ;;
        --storage)
            STORAGE_TYPE="$2"
            shift 2
            ;;
        --consul)
            CONSUL_ENABLED=true
            shift
            ;;
        --vault)
            VAULT_ENABLED=true
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

# Function to stop all distributed containers
stop_containers() {
    echo "Stopping all distributed containers..."
    docker rm -f anomaly_detection-haproxy 2>/dev/null || true
    docker rm -f anomaly_detection-consul 2>/dev/null || true
    docker rm -f anomaly_detection-vault 2>/dev/null || true

    for i in $(seq 1 $API_REPLICAS); do
        docker rm -f "anomaly_detection-api-${i}" 2>/dev/null || true
    done

    for i in $(seq 1 $WORKERS); do
        docker rm -f "anomaly_detection-worker-${i}" 2>/dev/null || true
    done

    # Storage containers
    docker rm -f anomaly_detection-postgres-master 2>/dev/null || true
    docker rm -f anomaly_detection-postgres-replica 2>/dev/null || true
    docker rm -f anomaly_detection-mongodb-primary 2>/dev/null || true
    docker rm -f anomaly_detection-mongodb-secondary1 2>/dev/null || true
    docker rm -f anomaly_detection-mongodb-secondary2 2>/dev/null || true
    docker rm -f anomaly_detection-redis-master 2>/dev/null || true
    docker rm -f anomaly_detection-redis-replica1 2>/dev/null || true
    docker rm -f anomaly_detection-redis-replica2 2>/dev/null || true

    echo "All distributed containers stopped."
    exit 0
}

if [[ "$STOP_CONTAINERS" == "true" ]]; then
    stop_containers
fi

# Create network
docker network create "$NETWORK_NAME" --driver bridge 2>/dev/null || true

# Clean up if requested
if [[ "$CLEAN_CONTAINERS" == "true" ]]; then
    echo "Cleaning up existing distributed containers..."
    stop_containers
fi

# Build production image if requested
if [[ "$BUILD_IMAGE" == "true" ]]; then
    echo "Building production image..."
    docker build -t "$IMAGE_NAME" -f "$PROJECT_ROOT/Dockerfile.hardened" "$PROJECT_ROOT"
fi

# Start service discovery with Consul
if [[ "$CONSUL_ENABLED" == "true" ]]; then
    echo "Starting Consul for service discovery..."
    docker run -d \
        --name anomaly_detection-consul \
        --network "$NETWORK_NAME" \
        -p 8500:8500 \
        -v consul-data:/consul/data \
        consul:latest agent -dev -client=0.0.0.0 -ui
fi

# Start secrets management with Vault
if [[ "$VAULT_ENABLED" == "true" ]]; then
    echo "Starting Vault for secrets management..."
    docker run -d \
        --name anomaly_detection-vault \
        --network "$NETWORK_NAME" \
        -p 8200:8200 \
        --cap-add=IPC_LOCK \
        -e VAULT_DEV_ROOT_TOKEN_ID=dev-root-token \
        -v vault-data:/vault/data \
        vault:latest
fi

# Start distributed storage based on type
case "$STORAGE_TYPE" in
    postgres)
        echo "Starting PostgreSQL cluster..."
        # Master
        docker run -d \
            --name anomaly_detection-postgres-master \
            --network "$NETWORK_NAME" \
            -e POSTGRES_DB=anomaly_detection \
            -e POSTGRES_USER=anomaly_detection \
            -e POSTGRES_PASSWORD=prod_password \
            -e POSTGRES_REPLICATION_MODE=master \
            -e POSTGRES_REPLICATION_USER=replicator \
            -e POSTGRES_REPLICATION_PASSWORD=replicator_password \
            -p 5432:5432 \
            -v postgres-master-data:/var/lib/postgresql/data \
            bitnami/postgresql:15

        # Replica
        docker run -d \
            --name anomaly_detection-postgres-replica \
            --network "$NETWORK_NAME" \
            -e POSTGRES_REPLICATION_MODE=slave \
            -e POSTGRES_REPLICATION_USER=replicator \
            -e POSTGRES_REPLICATION_PASSWORD=replicator_password \
            -e POSTGRES_MASTER_HOST=anomaly_detection-postgres-master \
            -e POSTGRES_MASTER_PORT_NUMBER=5432 \
            -p 5433:5432 \
            -v postgres-replica-data:/var/lib/postgresql/data \
            bitnami/postgresql:15
        ;;

    mongodb)
        echo "Starting MongoDB replica set..."
        # Primary
        docker run -d \
            --name anomaly_detection-mongodb-primary \
            --network "$NETWORK_NAME" \
            -e MONGODB_REPLICA_SET_MODE=primary \
            -e MONGODB_REPLICA_SET_NAME=rs0 \
            -e MONGODB_ROOT_PASSWORD=prod_password \
            -p 27017:27017 \
            -v mongodb-primary-data:/bitnami/mongodb \
            bitnami/mongodb:7.0

        # Secondary nodes
        for i in 1 2; do
            docker run -d \
                --name "anomaly_detection-mongodb-secondary${i}" \
                --network "$NETWORK_NAME" \
                -e MONGODB_REPLICA_SET_MODE=secondary \
                -e MONGODB_REPLICA_SET_NAME=rs0 \
                -e MONGODB_PRIMARY_HOST=anomaly_detection-mongodb-primary \
                -e MONGODB_PRIMARY_ROOT_PASSWORD=prod_password \
                -p "$((27017 + i)):27017" \
                -v "mongodb-secondary${i}-data:/bitnami/mongodb" \
                bitnami/mongodb:7.0
        done
        ;;

    redis-cluster)
        echo "Starting Redis cluster..."
        # Master
        docker run -d \
            --name anomaly_detection-redis-master \
            --network "$NETWORK_NAME" \
            -p 6379:6379 \
            -v redis-master-data:/data \
            redis:7-alpine redis-server --appendonly yes

        # Replicas
        for i in 1 2; do
            docker run -d \
                --name "anomaly_detection-redis-replica${i}" \
                --network "$NETWORK_NAME" \
                -p "$((6379 + i)):6379" \
                -v "redis-replica${i}-data:/data" \
                redis:7-alpine redis-server --appendonly yes --replicaof anomaly_detection-redis-master 6379
        done
        ;;
esac

# Wait for storage services to be ready
echo "Waiting for storage services to be ready..."
sleep 15

# Start API replicas
echo "Starting $API_REPLICAS API replicas..."
for i in $(seq 1 $API_REPLICAS); do
    API_PORT=$((8000 + i - 1))
    echo "Starting API replica $i on port $API_PORT..."

    docker run -d \
        --name "anomaly_detection-api-${i}" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        --env-file "$ENV_FILE" \
        -p "${API_PORT}:8000" \
        -v "${PROJECT_ROOT}/storage:/app/storage" \
        -v "${PROJECT_ROOT}/logs:/app/logs" \
        -e PYTHONPATH=/app/src \
        -e ENVIRONMENT=production \
        -e DEBUG=false \
        -e LOG_LEVEL=INFO \
        -e API_REPLICA_ID="$i" \
        -e STORAGE_TYPE="$STORAGE_TYPE" \
        --health-cmd "curl -f http://localhost:8000/health || exit 1" \
        --health-interval 30s \
        --health-timeout 10s \
        --health-retries 3 \
        "$IMAGE_NAME" \
        poetry run gunicorn anomaly_detection.presentation.api:app \
        --bind 0.0.0.0:8000 \
        --workers 2 \
        --worker-class uvicorn.workers.UvicornWorker \
        --worker-connections 500 \
        --max-requests 1000 \
        --timeout 30
done

# Start worker nodes
echo "Starting $WORKERS worker nodes..."
for i in $(seq 1 $WORKERS); do
    echo "Starting worker node $i..."

    docker run -d \
        --name "anomaly_detection-worker-${i}" \
        --network "$NETWORK_NAME" \
        --restart unless-stopped \
        --env-file "$ENV_FILE" \
        -v "${PROJECT_ROOT}/storage:/app/storage" \
        -v "${PROJECT_ROOT}/logs:/app/logs" \
        -e PYTHONPATH=/app/src \
        -e ENVIRONMENT=production \
        -e DEBUG=false \
        -e LOG_LEVEL=INFO \
        -e WORKER_ID="$i" \
        -e STORAGE_TYPE="$STORAGE_TYPE" \
        --memory 4g \
        --cpus 2.0 \
        "$IMAGE_NAME" \
        poetry run python -m anomaly_detection.infrastructure.distributed.worker \
        --worker-id "$i" \
        --concurrency 4
done

# Start HAProxy load balancer
echo "Starting HAProxy load balancer..."
cat > /tmp/haproxy.cfg << EOF
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog

frontend api_frontend
    bind *:80
    default_backend api_servers

backend api_servers
    balance roundrobin
    option httpchk GET /health
EOF

for i in $(seq 1 $API_REPLICAS); do
    API_PORT=$((8000 + i - 1))
    echo "    server api${i} anomaly_detection-api-${i}:8000 check" >> /tmp/haproxy.cfg
done

docker run -d \
    --name anomaly_detection-haproxy \
    --network "$NETWORK_NAME" \
    -p 80:80 \
    -v /tmp/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg \
    haproxy:2.8-alpine

echo "Distributed production deployment completed!"
echo "Load Balancer: http://localhost:80"
echo "API Replicas: $API_REPLICAS"
echo "Worker Nodes: $WORKERS"
echo "Storage: $STORAGE_TYPE"

if [[ "$CONSUL_ENABLED" == "true" ]]; then
    echo "Consul UI: http://localhost:8500"
fi
if [[ "$VAULT_ENABLED" == "true" ]]; then
    echo "Vault UI: http://localhost:8200"
fi

echo ""
echo "Use --stop to stop all containers"

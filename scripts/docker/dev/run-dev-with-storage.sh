#!/bin/bash

# Development Environment with Storage Infrastructure
# Includes PostgreSQL, Redis, and MinIO for complete development setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default configuration
ENV_FILE="${PROJECT_ROOT}/.env.dev"
NETWORK_NAME="pynomaly-dev"
CONTAINER_NAME="pynomaly-dev"
IMAGE_NAME="pynomaly:dev"
HOST_PORT=8000

# Storage configuration
POSTGRES_CONTAINER="pynomaly-postgres-dev"
REDIS_CONTAINER="pynomaly-redis-dev"
MINIO_CONTAINER="pynomaly-minio-dev"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -p, --port PORT      Host port to bind (default: 8000)"
    echo "  --storage TYPE       Storage backend (postgres|redis|minio|all) (default: all)"
    echo "  --build              Build image before running"
    echo "  --clean              Remove existing containers first"
    echo "  --stop               Stop all containers"
    echo "  -h, --help           Show this help message"
    exit 1
}

# Parse command line arguments
STORAGE_TYPE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            HOST_PORT="$2"
            shift 2
            ;;
        --storage)
            STORAGE_TYPE="$2"
            shift 2
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

# Function to stop all containers
stop_containers() {
    echo "Stopping all containers..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker rm -f "$POSTGRES_CONTAINER" 2>/dev/null || true
    docker rm -f "$REDIS_CONTAINER" 2>/dev/null || true
    docker rm -f "$MINIO_CONTAINER" 2>/dev/null || true
    echo "All containers stopped."
    exit 0
}

if [[ "$STOP_CONTAINERS" == "true" ]]; then
    stop_containers
fi

# Create network
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Clean up if requested
if [[ "$CLEAN_CONTAINERS" == "true" ]]; then
    echo "Cleaning up existing containers..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker rm -f "$POSTGRES_CONTAINER" 2>/dev/null || true
    docker rm -f "$REDIS_CONTAINER" 2>/dev/null || true
    docker rm -f "$MINIO_CONTAINER" 2>/dev/null || true
fi

# Start PostgreSQL if needed
if [[ "$STORAGE_TYPE" == "postgres" || "$STORAGE_TYPE" == "all" ]]; then
    echo "Starting PostgreSQL..."
    docker run -d \
        --name "$POSTGRES_CONTAINER" \
        --network "$NETWORK_NAME" \
        -e POSTGRES_DB=pynomaly_dev \
        -e POSTGRES_USER=pynomaly \
        -e POSTGRES_PASSWORD=dev_password \
        -p 5432:5432 \
        -v pynomaly-postgres-dev:/var/lib/postgresql/data \
        postgres:15-alpine
fi

# Start Redis if needed
if [[ "$STORAGE_TYPE" == "redis" || "$STORAGE_TYPE" == "all" ]]; then
    echo "Starting Redis..."
    docker run -d \
        --name "$REDIS_CONTAINER" \
        --network "$NETWORK_NAME" \
        -p 6379:6379 \
        -v pynomaly-redis-dev:/data \
        redis:7-alpine redis-server --appendonly yes
fi

# Start MinIO if needed
if [[ "$STORAGE_TYPE" == "minio" || "$STORAGE_TYPE" == "all" ]]; then
    echo "Starting MinIO..."
    docker run -d \
        --name "$MINIO_CONTAINER" \
        --network "$NETWORK_NAME" \
        -e MINIO_ROOT_USER=minioadmin \
        -e MINIO_ROOT_PASSWORD=minioadmin123 \
        -p 9000:9000 \
        -p 9001:9001 \
        -v pynomaly-minio-dev:/data \
        minio/minio server /data --console-address ":9001"
fi

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Build image if requested
if [[ "$BUILD_IMAGE" == "true" ]]; then
    echo "Building development image..."
    docker build -t "$IMAGE_NAME" -f "$PROJECT_ROOT/Dockerfile" "$PROJECT_ROOT"
fi

echo "Starting Pynomaly development environment with storage..."
echo "Container: $CONTAINER_NAME"
echo "Port: $HOST_PORT"
echo "Storage: $STORAGE_TYPE"

# Prepare environment variables
ENV_VARS=""
if [[ "$STORAGE_TYPE" == "postgres" || "$STORAGE_TYPE" == "all" ]]; then
    ENV_VARS="$ENV_VARS -e DATABASE_URL=postgresql://pynomaly:dev_password@$POSTGRES_CONTAINER:5432/pynomaly_dev"
fi
if [[ "$STORAGE_TYPE" == "redis" || "$STORAGE_TYPE" == "all" ]]; then
    ENV_VARS="$ENV_VARS -e REDIS_URL=redis://$REDIS_CONTAINER:6379/0"
fi
if [[ "$STORAGE_TYPE" == "minio" || "$STORAGE_TYPE" == "all" ]]; then
    ENV_VARS="$ENV_VARS -e MINIO_ENDPOINT=$MINIO_CONTAINER:9000"
    ENV_VARS="$ENV_VARS -e MINIO_ACCESS_KEY=minioadmin"
    ENV_VARS="$ENV_VARS -e MINIO_SECRET_KEY=minioadmin123"
fi

# Run the main application container
docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --network "$NETWORK_NAME" \
    --env-file "$ENV_FILE" \
    $ENV_VARS \
    -p "${HOST_PORT}:8000" \
    -v "${PROJECT_ROOT}:/app" \
    -v "${PROJECT_ROOT}/.venv:/app/.venv" \
    -v "${PROJECT_ROOT}/storage:/app/storage" \
    -e PYTHONPATH=/app/src \
    -e ENVIRONMENT=development \
    -e DEBUG=true \
    -e LOG_LEVEL=DEBUG \
    -e RELOAD=true \
    "$IMAGE_NAME" \
    poetry run uvicorn pynomaly.presentation.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir /app/src

echo "Development environment stopped."
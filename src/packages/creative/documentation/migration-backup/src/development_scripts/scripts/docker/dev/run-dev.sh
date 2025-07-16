#!/bin/bash

# Development Environment - Docker Run Scripts
# Basic development setup with hot-reload and dev tools

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default configuration
ENV_FILE="${PROJECT_ROOT}/.env.dev"
NETWORK_NAME="pynomaly-dev"
CONTAINER_NAME="pynomaly-dev"
IMAGE_NAME="pynomaly:dev"
HOST_PORT=8000
CONTAINER_PORT=8000

# Load environment variables if file exists
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
fi

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -p, --port PORT      Host port to bind (default: 8000)"
    echo "  -n, --name NAME      Container name (default: pynomaly-dev)"
    echo "  -i, --image IMAGE    Docker image name (default: pynomaly:dev)"
    echo "  --env-file FILE      Environment file path (default: .env.dev)"
    echo "  --build              Build image before running"
    echo "  --clean              Remove existing container first"
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
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --build)
            BUILD_IMAGE=true
            shift
            ;;
        --clean)
            CLEAN_CONTAINER=true
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

# Create network if it doesn't exist
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Clean up existing container if requested
if [[ "$CLEAN_CONTAINER" == "true" ]]; then
    echo "Cleaning up existing container..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
fi

# Build image if requested
if [[ "$BUILD_IMAGE" == "true" ]]; then
    echo "Building development image..."
    docker build -t "$IMAGE_NAME" -f "$PROJECT_ROOT/Dockerfile" "$PROJECT_ROOT"
fi

echo "Starting Pynomaly development environment..."
echo "Container: $CONTAINER_NAME"
echo "Image: $IMAGE_NAME"
echo "Port: $HOST_PORT -> $CONTAINER_PORT"
echo "Network: $NETWORK_NAME"

# Run the container
docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --network "$NETWORK_NAME" \
    --env-file "$ENV_FILE" \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
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
    --port "$CONTAINER_PORT" \
    --reload \
    --reload-dir /app/src

echo "Development server stopped."

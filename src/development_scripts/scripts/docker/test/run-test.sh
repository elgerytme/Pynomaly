#!/bin/bash

# Test Environment - Docker Run Scripts
# Isolated test environment with test database and services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default configuration
ENV_FILE="${PROJECT_ROOT}/.env.test"
NETWORK_NAME="anomaly_detection-test"
CONTAINER_NAME="anomaly_detection-test"
IMAGE_NAME="anomaly_detection:test"
TEST_TYPE="unit"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -t, --type TYPE      Test type (unit|integration|e2e|all) (default: unit)"
    echo "  -c, --coverage       Run with coverage reporting"
    echo "  --parallel           Run tests in parallel"
    echo "  --build              Build test image before running"
    echo "  --clean              Remove existing containers first"
    echo "  --storage TYPE       Include storage services (postgres|redis|all)"
    echo "  -h, --help           Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
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
        --storage)
            STORAGE_TYPE="$2"
            shift 2
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

# Create network
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Clean up if requested
if [[ "$CLEAN_CONTAINERS" == "true" ]]; then
    echo "Cleaning up existing test containers..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker rm -f "anomaly_detection-postgres-test" 2>/dev/null || true
    docker rm -f "anomaly_detection-redis-test" 2>/dev/null || true
fi

# Start test storage services if needed
if [[ -n "$STORAGE_TYPE" ]]; then
    if [[ "$STORAGE_TYPE" == "postgres" || "$STORAGE_TYPE" == "all" ]]; then
        echo "Starting test PostgreSQL..."
        docker run -d \
            --name "anomaly_detection-postgres-test" \
            --network "$NETWORK_NAME" \
            -e POSTGRES_DB=anomaly_detection_test \
            -e POSTGRES_USER=test_user \
            -e POSTGRES_PASSWORD=test_password \
            -p 5433:5432 \
            postgres:15-alpine
    fi

    if [[ "$STORAGE_TYPE" == "redis" || "$STORAGE_TYPE" == "all" ]]; then
        echo "Starting test Redis..."
        docker run -d \
            --name "anomaly_detection-redis-test" \
            --network "$NETWORK_NAME" \
            -p 6380:6379 \
            redis:7-alpine
    fi

    # Wait for services
    echo "Waiting for test services to be ready..."
    sleep 5
fi

# Build test image if requested
if [[ "$BUILD_IMAGE" == "true" ]]; then
    echo "Building test image..."
    docker build -t "$IMAGE_NAME" -f "$PROJECT_ROOT/Dockerfile" "$PROJECT_ROOT"
fi

# Prepare test command
TEST_CMD="poetry run pytest"

case "$TEST_TYPE" in
    unit)
        TEST_CMD="$TEST_CMD tests/unit tests/domain"
        ;;
    integration)
        TEST_CMD="$TEST_CMD tests/integration tests/infrastructure"
        ;;
    e2e)
        TEST_CMD="$TEST_CMD tests/e2e tests/presentation"
        ;;
    all)
        TEST_CMD="$TEST_CMD"
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        exit 1
        ;;
esac

# Add coverage if requested
if [[ "$COVERAGE" == "true" ]]; then
    TEST_CMD="$TEST_CMD --cov=anomaly_detection --cov-report=html --cov-report=xml --cov-report=term"
fi

# Add parallel execution if requested
if [[ "$PARALLEL" == "true" ]]; then
    TEST_CMD="$TEST_CMD -n auto"
fi

# Add test-specific options
TEST_CMD="$TEST_CMD -v --tb=short --strict-markers"

echo "Running tests: $TEST_TYPE"
echo "Command: $TEST_CMD"

# Prepare environment variables
ENV_VARS=""
if [[ "$STORAGE_TYPE" == "postgres" || "$STORAGE_TYPE" == "all" ]]; then
    ENV_VARS="$ENV_VARS -e DATABASE_URL=postgresql://test_user:test_password@anomaly_detection-postgres-test:5432/anomaly_detection_test"
fi
if [[ "$STORAGE_TYPE" == "redis" || "$STORAGE_TYPE" == "all" ]]; then
    ENV_VARS="$ENV_VARS -e REDIS_URL=redis://anomaly_detection-redis-test:6379/0"
fi

# Run tests in container
docker run --rm \
    --name "$CONTAINER_NAME" \
    --network "$NETWORK_NAME" \
    --env-file "$ENV_FILE" \
    $ENV_VARS \
    -v "${PROJECT_ROOT}:/app" \
    -v "${PROJECT_ROOT}/htmlcov:/app/htmlcov" \
    -v "${PROJECT_ROOT}/coverage.xml:/app/coverage.xml" \
    -e PYTHONPATH=/app/src \
    -e ENVIRONMENT=test \
    -e DEBUG=false \
    -e LOG_LEVEL=WARNING \
    "$IMAGE_NAME" \
    bash -c "$TEST_CMD"

echo "Tests completed."

# Clean up test storage services
if [[ -n "$STORAGE_TYPE" ]]; then
    echo "Cleaning up test services..."
    docker rm -f "anomaly_detection-postgres-test" 2>/dev/null || true
    docker rm -f "anomaly_detection-redis-test" 2>/dev/null || true
fi

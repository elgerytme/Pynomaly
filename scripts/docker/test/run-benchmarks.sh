#!/bin/bash

# Benchmark Testing Environment
# Performance and load testing with monitoring

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default configuration
NETWORK_NAME="pynomaly-benchmark"
CONTAINER_NAME="pynomaly-benchmark"
IMAGE_NAME="pynomaly:benchmark"
BENCHMARK_TYPE="performance"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -t, --type TYPE      Benchmark type (performance|load|stress|memory) (default: performance)"
    echo "  --datasets PATH      Custom datasets directory"
    echo "  --algorithms LIST    Comma-separated algorithm list"
    echo "  --build              Build benchmark image before running"
    echo "  --clean              Remove existing containers first"
    echo "  --monitoring         Include monitoring stack (Prometheus/Grafana)"
    echo "  -h, --help           Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BENCHMARK_TYPE="$2"
            shift 2
            ;;
        --datasets)
            DATASETS_PATH="$2"
            shift 2
            ;;
        --algorithms)
            ALGORITHMS="$2"
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
        --monitoring)
            MONITORING=true
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

# Create network
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Clean up if requested
if [[ "$CLEAN_CONTAINERS" == "true" ]]; then
    echo "Cleaning up existing benchmark containers..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker rm -f "pynomaly-prometheus" 2>/dev/null || true
    docker rm -f "pynomaly-grafana" 2>/dev/null || true
fi

# Start monitoring stack if requested
if [[ "$MONITORING" == "true" ]]; then
    echo "Starting monitoring stack..."
    
    # Start Prometheus
    docker run -d \
        --name "pynomaly-prometheus" \
        --network "$NETWORK_NAME" \
        -p 9090:9090 \
        -v "${PROJECT_ROOT}/docker/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml" \
        prom/prometheus:latest

    # Start Grafana
    docker run -d \
        --name "pynomaly-grafana" \
        --network "$NETWORK_NAME" \
        -p 3000:3000 \
        -e GF_SECURITY_ADMIN_PASSWORD=admin \
        -v grafana-storage:/var/lib/grafana \
        grafana/grafana:latest

    echo "Monitoring available at: http://localhost:3000 (admin/admin)"
fi

# Build benchmark image if requested
if [[ "$BUILD_IMAGE" == "true" ]]; then
    echo "Building benchmark image..."
    docker build -t "$IMAGE_NAME" -f "$PROJECT_ROOT/Dockerfile" "$PROJECT_ROOT"
fi

# Prepare benchmark command based on type
case "$BENCHMARK_TYPE" in
    performance)
        BENCH_CMD="poetry run python -m benchmarks.performance"
        ;;
    load)
        BENCH_CMD="poetry run python -m benchmarks.load_testing"
        ;;
    stress)
        BENCH_CMD="poetry run python -m benchmarks.stress_testing"
        ;;
    memory)
        BENCH_CMD="poetry run python -m benchmarks.memory_profiling"
        ;;
    *)
        echo "Unknown benchmark type: $BENCHMARK_TYPE"
        exit 1
        ;;
esac

# Add algorithm filter if specified
if [[ -n "$ALGORITHMS" ]]; then
    BENCH_CMD="$BENCH_CMD --algorithms $ALGORITHMS"
fi

# Add custom datasets if specified
DATASET_MOUNT=""
if [[ -n "$DATASETS_PATH" ]]; then
    DATASET_MOUNT="-v ${DATASETS_PATH}:/app/benchmark_datasets"
    BENCH_CMD="$BENCH_CMD --datasets /app/benchmark_datasets"
fi

echo "Running benchmarks: $BENCHMARK_TYPE"
echo "Command: $BENCH_CMD"

# Run benchmarks in container
docker run --rm \
    --name "$CONTAINER_NAME" \
    --network "$NETWORK_NAME" \
    -v "${PROJECT_ROOT}:/app" \
    -v "${PROJECT_ROOT}/benchmarks/results:/app/benchmarks/results" \
    $DATASET_MOUNT \
    -e PYTHONPATH=/app/src \
    -e ENVIRONMENT=benchmark \
    -e LOG_LEVEL=INFO \
    -e BENCHMARK_TYPE="$BENCHMARK_TYPE" \
    --cpus="0.000" \
    --memory="4g" \
    "$IMAGE_NAME" \
    bash -c "$BENCH_CMD"

echo "Benchmarks completed. Results saved to benchmarks/results/"

# Keep monitoring stack running if requested
if [[ "$MONITORING" == "true" ]]; then
    echo "Monitoring stack is still running. Use --clean to stop."
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000"
fi
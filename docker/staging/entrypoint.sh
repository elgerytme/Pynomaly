#!/bin/bash

# Pynomaly Staging Environment Entrypoint
# This script prepares the staging environment and starts the application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Environment validation
validate_environment() {
    log_info "Validating staging environment..."

    # Check required environment variables
    local required_vars=(
        "PYNOMALY_ENV"
        "DATABASE_URL"
        "REDIS_URL"
        "MONGODB_URL"
    )

    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done

    # Validate environment is staging
    if [[ "$PYNOMALY_ENV" != "staging" ]]; then
        log_error "Environment must be 'staging', got: $PYNOMALY_ENV"
        exit 1
    fi

    log_success "Environment validation passed"
}

# Wait for database services
wait_for_databases() {
    log_info "Waiting for database services..."

    # Extract database connection details
    local postgres_host=$(echo "$DATABASE_URL" | grep -oP '(?<=@)[^:/]+' | head -1)
    local postgres_port=$(echo "$DATABASE_URL" | grep -oP ':\d+' | tr -d ':' | head -1)
    local redis_host=$(echo "$REDIS_URL" | grep -oP '(?<=@)[^:/]+' | head -1)
    local redis_port=$(echo "$REDIS_URL" | grep -oP ':\d+' | tr -d ':' | head -1)
    local mongodb_host=$(echo "$MONGODB_URL" | grep -oP '(?<=@)[^:/]+' | head -1)
    local mongodb_port=$(echo "$MONGODB_URL" | grep -oP ':\d+' | tr -d ':' | head -1)

    # Wait for PostgreSQL
    if [[ -n "$postgres_host" && -n "$postgres_port" ]]; then
        log_info "Waiting for PostgreSQL at $postgres_host:$postgres_port..."
        ./wait-for-it.sh "$postgres_host:$postgres_port" --timeout=60 --strict
        log_success "PostgreSQL is ready"
    fi

    # Wait for Redis
    if [[ -n "$redis_host" && -n "$redis_port" ]]; then
        log_info "Waiting for Redis at $redis_host:$redis_port..."
        ./wait-for-it.sh "$redis_host:$redis_port" --timeout=60 --strict
        log_success "Redis is ready"
    fi

    # Wait for MongoDB
    if [[ -n "$mongodb_host" && -n "$mongodb_port" ]]; then
        log_info "Waiting for MongoDB at $mongodb_host:$mongodb_port..."
        ./wait-for-it.sh "$mongodb_host:$mongodb_port" --timeout=60 --strict
        log_success "MongoDB is ready"
    fi

    log_success "All database services are ready"
}

# Initialize database
initialize_database() {
    log_info "Initializing database..."

    # Run database migrations
    if command -v alembic &> /dev/null; then
        log_info "Running database migrations..."
        alembic upgrade head
        log_success "Database migrations completed"
    else
        log_warning "Alembic not available, skipping migrations"
    fi

    # Seed test data for staging
    if [[ "$LOAD_TEST_MODE" == "true" ]]; then
        log_info "Seeding test data for staging..."
        python -c "
import sys
sys.path.insert(0, '/app/src')
try:
    from pynomaly.infrastructure.database.seed_data import seed_staging_data
    seed_staging_data()
    print('Test data seeded successfully')
except ImportError as e:
    print(f'Warning: Could not seed test data: {e}')
except Exception as e:
    print(f'Error seeding test data: {e}')
"
        log_success "Test data seeding completed"
    fi
}

# Setup logging
setup_logging() {
    log_info "Setting up logging..."

    # Create log directories
    mkdir -p /app/logs

    # Set log file permissions
    touch /app/logs/pynomaly.log
    touch /app/logs/access.log
    touch /app/logs/error.log

    # Configure log rotation (simple version)
    if [[ -f /app/logs/pynomaly.log ]] && [[ $(stat -c%s /app/logs/pynomaly.log) -gt 100000000 ]]; then
        log_info "Rotating log files..."
        mv /app/logs/pynomaly.log /app/logs/pynomaly.log.$(date +%Y%m%d_%H%M%S)
        touch /app/logs/pynomaly.log
    fi

    log_success "Logging setup completed"
}

# Load test data
load_test_data() {
    if [[ "$LOAD_TEST_MODE" == "true" ]]; then
        log_info "Loading test data for staging..."

        # Create test data directory
        mkdir -p /app/test-data

        # Generate sample datasets if they don't exist
        if [[ ! -f /app/test-data/sample_data.json ]]; then
            log_info "Generating sample test data..."
            python -c "
import sys
sys.path.insert(0, '/app/src')
import json
import numpy as np
from datetime import datetime, timedelta

# Generate sample anomaly detection data
np.random.seed(42)
timestamps = [datetime.now() - timedelta(hours=i) for i in range(1000)]
normal_data = np.random.normal(0, 1, 900)
anomaly_data = np.random.normal(5, 0.5, 100)
all_data = np.concatenate([normal_data, anomaly_data])
np.random.shuffle(all_data)

test_data = {
    'dataset_id': 'staging_test_data',
    'data': [
        {
            'timestamp': timestamps[i].isoformat(),
            'value': float(all_data[i]),
            'is_anomaly': bool(all_data[i] > 3)
        }
        for i in range(len(all_data))
    ]
}

with open('/app/test-data/sample_data.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print('Sample test data generated')
"
        fi

        log_success "Test data loading completed"
    fi
}

# Performance monitoring setup
setup_monitoring() {
    log_info "Setting up performance monitoring..."

    # Enable Prometheus metrics
    export PROMETHEUS_ENABLED=true
    export METRICS_ENDPOINT="/metrics"

    # Set up application performance monitoring
    if [[ "$ENABLE_DEBUG_ENDPOINTS" == "true" ]]; then
        log_info "Debug endpoints enabled for staging"
        export DEBUG_ENDPOINTS_ENABLED=true
    fi

    log_success "Performance monitoring setup completed"
}

# Health check setup
setup_health_checks() {
    log_info "Setting up health checks..."

    # Ensure health check script is executable
    chmod +x ./healthcheck.sh

    # Set health check environment variables
    export HEALTH_CHECK_ENDPOINT="/health"
    export HEALTH_CHECK_TIMEOUT=30

    log_success "Health checks setup completed"
}

# Start application
start_application() {
    log_info "Starting Pynomaly application in staging mode..."

    # Display startup information
    log_info "Startup Configuration:"
    echo "  Environment: $PYNOMALY_ENV"
    echo "  Debug Mode: ${PYNOMALY_DEBUG:-false}"
    echo "  Log Level: ${PYNOMALY_LOG_LEVEL:-INFO}"
    echo "  Host: ${PYNOMALY_API_HOST:-0.0.0.0}"
    echo "  Port: ${PYNOMALY_API_PORT:-8000}"
    echo "  Prometheus: ${PROMETHEUS_ENABLED:-false}"
    echo "  Load Test Mode: ${LOAD_TEST_MODE:-false}"
    echo "  Swagger UI: ${ENABLE_SWAGGER:-false}"
    echo "  Debug Endpoints: ${ENABLE_DEBUG_ENDPOINTS:-false}"

    # Start the application
    log_success "Starting application with command: $*"
    exec "$@"
}

# Signal handlers
handle_sigterm() {
    log_info "Received SIGTERM, shutting down gracefully..."
    # Add any cleanup code here
    exit 0
}

handle_sigint() {
    log_info "Received SIGINT, shutting down gracefully..."
    # Add any cleanup code here
    exit 0
}

# Set up signal handlers
trap handle_sigterm SIGTERM
trap handle_sigint SIGINT

# Main execution
main() {
    log_info "Starting Pynomaly staging environment..."

    # Run initialization steps
    validate_environment
    wait_for_databases
    setup_logging
    initialize_database
    load_test_data
    setup_monitoring
    setup_health_checks

    # Start the application
    start_application "$@"
}

# Run main function with all arguments
main "$@"

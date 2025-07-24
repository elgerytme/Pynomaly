#!/bin/bash
# Production entrypoint script for anomaly detection service

set -e

# Default values
MODE=${1:-api}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-4}
LOG_LEVEL=${LOG_LEVEL:-info}

# Ensure log directory exists
mkdir -p logs

# Function to wait for dependencies
wait_for_dependencies() {
    echo "Waiting for dependencies..."
    
    # Wait for database if configured
    if [ -n "$DATABASE_URL" ]; then
        echo "Waiting for database..."
        python -c "
import time
import sys
import psycopg2
from urllib.parse import urlparse

url = '$DATABASE_URL'
parsed = urlparse(url)

max_attempts = 30
attempt = 0

while attempt < max_attempts:
    try:
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            database=parsed.path[1:]
        )
        conn.close()
        print('Database connection successful')
        break
    except Exception as e:
        attempt += 1
        print(f'Database connection attempt {attempt}/{max_attempts} failed: {e}')
        time.sleep(2)
else:
    print('Failed to connect to database after all attempts')
    sys.exit(1)
"
    fi
    
    # Wait for Redis if configured
    if [ -n "$REDIS_URL" ]; then
        echo "Waiting for Redis..."
        python -c "
import time
import sys
import redis
from urllib.parse import urlparse

url = '$REDIS_URL'
parsed = urlparse(url)

max_attempts = 30
attempt = 0

while attempt < max_attempts:
    try:
        r = redis.Redis(
            host=parsed.hostname,
            port=parsed.port or 6379,
            password=parsed.password,
            decode_responses=True
        )
        r.ping()
        print('Redis connection successful')
        break
    except Exception as e:
        attempt += 1
        print(f'Redis connection attempt {attempt}/{max_attempts} failed: {e}')
        time.sleep(2)
else:
    print('Failed to connect to Redis after all attempts')
    sys.exit(1)
"
    fi
}

# Function to run database migrations
run_migrations() {
    echo "Running database migrations..."
    python -m anomaly_detection.infrastructure.database.migrations.run_migrations
}

# Function to initialize system
initialize_system() {
    echo "Initializing anomaly detection system..."
    
    # Create necessary directories
    mkdir -p data/models data/cache data/uploads logs/app logs/access
    
    # Set proper permissions
    chmod 755 data logs
    chmod 644 config/*.yml config/*.json 2>/dev/null || true
    
    # Initialize default models if needed
    if [ "$INIT_DEFAULT_MODELS" = "true" ]; then
        echo "Initializing default models..."
        python -c "
from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository
from anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter import SklearnAdapter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    repo = ModelRepository()
    
    # Initialize default Isolation Forest model
    adapter = SklearnAdapter('iforest', contamination=0.1, n_estimators=100)
    repo.save_model('default_isolation_forest', adapter, {
        'algorithm': 'iforest',
        'description': 'Default Isolation Forest model',
        'version': '1.0.0'
    })
    
    logger.info('Default models initialized successfully')
except Exception as e:
    logger.error(f'Failed to initialize default models: {e}')
"
    fi
}

# Function to start API server
start_api() {
    echo "Starting API server on $HOST:$PORT with $WORKERS workers..."
    exec uvicorn anomaly_detection.api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL" \
        --access-log \
        --access-logfile logs/access.log \
        --log-config config/logging.yml
}

# Function to start web interface
start_web() {
    echo "Starting web interface on $HOST:${WEB_PORT:-8080}..."
    exec uvicorn anomaly_detection.web.main:app \
        --host "$HOST" \
        --port "${WEB_PORT:-8080}" \
        --workers "${WEB_WORKERS:-2}" \
        --log-level "$LOG_LEVEL"
}

# Function to start worker
start_worker() {
    echo "Starting background worker..."
    exec python -m anomaly_detection.worker \
        --log-level "$LOG_LEVEL" \
        --concurrency "${WORKER_CONCURRENCY:-4}"
}

# Function to start scheduler
start_scheduler() {
    echo "Starting task scheduler..."
    exec python -m anomaly_detection.scheduler \
        --log-level "$LOG_LEVEL"
}

# Function to run CLI
run_cli() {
    echo "Running CLI command: ${*:2}"
    exec python -m anomaly_detection.cli "${@:2}"
}

# Function to run health check
run_healthcheck() {
    echo "Running health check..."
    exec python healthcheck.py "$@"
}

# Function to run tests
run_tests() {
    echo "Running tests..."
    exec pytest tests/ \
        --verbose \
        --cov=anomaly_detection \
        --cov-report=html:logs/coverage \
        --cov-report=term \
        --junit-xml=logs/test-results.xml
}

# Function to start monitoring
start_monitoring() {
    echo "Starting monitoring services..."
    # Start Prometheus metrics endpoint
    python -m anomaly_detection.infrastructure.monitoring.prometheus_metrics &
    
    # Start health monitoring
    python -m anomaly_detection.infrastructure.monitoring.health_monitor &
    
    wait
}

# Main execution logic
case "$MODE" in
    "api")
        wait_for_dependencies
        run_migrations
        initialize_system
        start_api
        ;;
    "web")
        wait_for_dependencies
        initialize_system
        start_web
        ;;
    "worker")
        wait_for_dependencies
        start_worker
        ;;
    "scheduler")
        wait_for_dependencies
        start_scheduler
        ;;
    "cli")
        run_cli "$@"
        ;;
    "healthcheck")
        run_healthcheck "${@:2}"
        ;;
    "test")
        run_tests
        ;;
    "monitoring")
        start_monitoring
        ;;
    "all")
        echo "Starting all services..."
        wait_for_dependencies
        run_migrations
        initialize_system
        
        # Start API server in background
        start_api &
        API_PID=$!
        
        # Start web interface in background
        start_web &
        WEB_PID=$!
        
        # Start worker in background
        start_worker &
        WORKER_PID=$!
        
        # Start monitoring
        start_monitoring &
        MONITOR_PID=$!
        
        # Wait for any service to exit
        wait -n $API_PID $WEB_PID $WORKER_PID $MONITOR_PID
        
        # Kill remaining services
        kill $API_PID $WEB_PID $WORKER_PID $MONITOR_PID 2>/dev/null || true
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: api, web, worker, scheduler, cli, healthcheck, test, monitoring, all"
        exit 1
        ;;
esac
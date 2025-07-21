#!/bin/bash
# Entrypoint script for Pynomaly Detection container

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Banner
cat << 'EOF'
🚀 Pynomaly Detection - Production Container
============================================
Advanced Anomaly Detection for Enterprise
============================================
EOF

# Environment validation
log "Validating environment..."

# Set default values
export PYNOMALY_ENV=${PYNOMALY_ENV:-production}
export PYNOMALY_LOG_LEVEL=${PYNOMALY_LOG_LEVEL:-INFO}
export PYNOMALY_HOST=${PYNOMALY_HOST:-0.0.0.0}
export PYNOMALY_PORT=${PYNOMALY_PORT:-8080}
export PYNOMALY_WORKERS=${PYNOMALY_WORKERS:-4}
export PYNOMALY_MAX_REQUESTS=${PYNOMALY_MAX_REQUESTS:-1000}
export PYNOMALY_TIMEOUT=${PYNOMALY_TIMEOUT:-30}
export PYNOMALY_MEMORY_LIMIT_GB=${PYNOMALY_MEMORY_LIMIT_GB:-4}

# Display configuration
log "Configuration:"
echo "  Environment: ${PYNOMALY_ENV}"
echo "  Log Level: ${PYNOMALY_LOG_LEVEL}"
echo "  Host: ${PYNOMALY_HOST}"
echo "  Port: ${PYNOMALY_PORT}"
echo "  Workers: ${PYNOMALY_WORKERS}"
echo "  Memory Limit: ${PYNOMALY_MEMORY_LIMIT_GB}GB"

# System checks
log "Performing system checks..."

# Check Python installation
if ! python --version > /dev/null 2>&1; then
    error "Python is not available"
    exit 1
fi

# Check Pynomaly installation
if ! python -c "import pynomaly_detection" > /dev/null 2>&1; then
    error "Pynomaly Detection is not properly installed"
    exit 1
fi

# Check Phase 2 availability
log "Checking Phase 2 component availability..."
python -c "
from pynomaly_detection import check_phase2_availability
availability = check_phase2_availability()
all_available = all(availability.values())
print(f'Phase 2 Available: {all_available}')
if not all_available:
    print(f'Missing components: {[k for k, v in availability.items() if not v]}')
    exit(1)
" || {
    error "Phase 2 components not fully available"
    exit 1
}

success "System checks passed"

# Memory optimization
log "Configuring memory optimization..."
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_TOP_PAD_=131072
export MALLOC_MMAP_MAX_=65536

# Set Python optimization
export PYTHONOPTIMIZE=1

# Initialize data directories
log "Initializing data directories..."
mkdir -p /app/data/{models,cache,logs,metrics}
mkdir -p /app/config/{detection,monitoring}

# Copy default configurations if they don't exist
if [ ! -f "/app/config/detection.yaml" ]; then
    log "Setting up default detection configuration..."
    cp /app/config/default-detection.yaml /app/config/detection.yaml 2>/dev/null || true
fi

if [ ! -f "/app/config/monitoring.yaml" ]; then
    log "Setting up default monitoring configuration..."
    cp /app/config/default-monitoring.yaml /app/config/monitoring.yaml 2>/dev/null || true
fi

# Command handling
case "${1}" in
    server)
        log "Starting Pynomaly Detection API server..."
        exec python -m pynomaly_detection.server \
            --host "${PYNOMALY_HOST}" \
            --port "${PYNOMALY_PORT}" \
            --workers "${PYNOMALY_WORKERS}" \
            --max-requests "${PYNOMALY_MAX_REQUESTS}" \
            --timeout "${PYNOMALY_TIMEOUT}" \
            --log-level "${PYNOMALY_LOG_LEVEL}"
        ;;
    
    worker)
        log "Starting Pynomaly Detection background worker..."
        exec python -m pynomaly_detection.worker \
            --log-level "${PYNOMALY_LOG_LEVEL}"
        ;;
    
    benchmark)
        log "Running performance benchmarks..."
        exec python -m pynomaly_detection.benchmark \
            --output-dir "/app/data/benchmarks" \
            "${@:2}"
        ;;
    
    migrate)
        log "Running database migrations..."
        exec python -m pynomaly_detection.migrate "${@:2}"
        ;;
    
    shell)
        log "Starting interactive shell..."
        exec python -c "
import pynomaly_detection
print('Pynomaly Detection shell - Phase 2 available')
print('Available components:')
from pynomaly_detection import check_phase2_availability
for component, available in check_phase2_availability().items():
    status = '✅' if available else '❌'
    print(f'  {status} {component}')
print()
print('Example usage:')
print('  from pynomaly_detection import CoreDetectionService')
print('  detector = CoreDetectionService()')
print()
"
        exec /bin/bash
        ;;
    
    test)
        log "Running tests..."
        exec python -m pytest /app/tests/ \
            --cov=pynomaly_detection \
            --cov-report=html:/app/data/coverage \
            --junit-xml=/app/data/test-results.xml \
            "${@:2}"
        ;;
    
    health)
        log "Performing health check..."
        exec python healthcheck.py
        ;;
    
    version)
        log "Version information:"
        python -c "
from pynomaly_detection import get_version_info
import json
info = get_version_info()
print(json.dumps(info, indent=2))
"
        ;;
    
    *)
        if [ -n "${1}" ]; then
            log "Executing custom command: ${*}"
            exec "${@}"
        else
            error "No command specified. Available commands:"
            echo "  server    - Start API server (default)"
            echo "  worker    - Start background worker"
            echo "  benchmark - Run performance benchmarks"
            echo "  migrate   - Run database migrations"
            echo "  shell     - Interactive shell"
            echo "  test      - Run test suite"
            echo "  health    - Health check"
            echo "  version   - Show version info"
            exit 1
        fi
        ;;
esac
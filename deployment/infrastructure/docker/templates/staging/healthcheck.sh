#!/bin/bash

# Pynomaly Staging Health Check Script
# This script performs comprehensive health checks for the staging environment

set -e

# Configuration
HEALTH_ENDPOINT="${HEALTH_CHECK_ENDPOINT:-/health}"
API_HOST="${PYNOMALY_API_HOST:-localhost}"
API_PORT="${PYNOMALY_API_PORT:-8000}"
TIMEOUT="${HEALTH_CHECK_TIMEOUT:-30}"
MAX_RETRIES=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[HEALTH]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[HEALTH]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[HEALTH]${NC} $1"
}

log_error() {
    echo -e "${RED}[HEALTH]${NC} $1"
}

# Check if application is responding
check_application_health() {
    local url="http://${API_HOST}:${API_PORT}${HEALTH_ENDPOINT}"
    local retry_count=0

    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        if curl -f -s --max-time "$TIMEOUT" "$url" > /dev/null 2>&1; then
            log_success "Application health check passed"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [[ $retry_count -lt $MAX_RETRIES ]]; then
                log_warning "Health check failed, retrying ($retry_count/$MAX_RETRIES)..."
                sleep 2
            else
                log_error "Application health check failed after $MAX_RETRIES attempts"
                return 1
            fi
        fi
    done
}

# Check database connectivity
check_database_health() {
    log_info "Checking database connectivity..."

    # Check PostgreSQL
    if [[ -n "${DATABASE_URL:-}" ]]; then
        local postgres_check=$(python3 -c "
import sys
sys.path.insert(0, '/app/src')
try:
    import psycopg2
    from urllib.parse import urlparse

    url = '$DATABASE_URL'
    parsed = urlparse(url)

    conn = psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port,
        database=parsed.path[1:],
        user=parsed.username,
        password=parsed.password
    )
    cursor = conn.cursor()
    cursor.execute('SELECT 1')
    cursor.close()
    conn.close()
    print('PostgreSQL: OK')
except Exception as e:
    print(f'PostgreSQL: ERROR - {e}')
    sys.exit(1)
" 2>&1)

        if [[ $? -eq 0 ]]; then
            log_success "PostgreSQL connectivity check passed"
        else
            log_error "PostgreSQL connectivity check failed: $postgres_check"
            return 1
        fi
    fi

    # Check Redis
    if [[ -n "${REDIS_URL:-}" ]]; then
        local redis_check=$(python3 -c "
import sys
sys.path.insert(0, '/app/src')
try:
    import redis
    from urllib.parse import urlparse

    url = '$REDIS_URL'
    parsed = urlparse(url)

    r = redis.Redis(
        host=parsed.hostname,
        port=parsed.port,
        password=parsed.password,
        decode_responses=True
    )
    r.ping()
    print('Redis: OK')
except Exception as e:
    print(f'Redis: ERROR - {e}')
    sys.exit(1)
" 2>&1)

        if [[ $? -eq 0 ]]; then
            log_success "Redis connectivity check passed"
        else
            log_error "Redis connectivity check failed: $redis_check"
            return 1
        fi
    fi

    # Check MongoDB
    if [[ -n "${MONGODB_URL:-}" ]]; then
        local mongodb_check=$(python3 -c "
import sys
sys.path.insert(0, '/app/src')
try:
    from pymongo import MongoClient
    from urllib.parse import urlparse

    url = '$MONGODB_URL'
    client = MongoClient(url, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print('MongoDB: OK')
except Exception as e:
    print(f'MongoDB: ERROR - {e}')
    sys.exit(1)
" 2>&1)

        if [[ $? -eq 0 ]]; then
            log_success "MongoDB connectivity check passed"
        else
            log_error "MongoDB connectivity check failed: $mongodb_check"
            return 1
        fi
    fi

    return 0
}

# Check memory usage
check_memory_usage() {
    local memory_info=$(cat /proc/meminfo)
    local total_memory=$(echo "$memory_info" | grep MemTotal | awk '{print $2}')
    local available_memory=$(echo "$memory_info" | grep MemAvailable | awk '{print $2}')

    if [[ -n "$total_memory" && -n "$available_memory" ]]; then
        local memory_usage_percent=$(( (total_memory - available_memory) * 100 / total_memory ))

        if [[ $memory_usage_percent -gt 90 ]]; then
            log_error "High memory usage: ${memory_usage_percent}%"
            return 1
        elif [[ $memory_usage_percent -gt 80 ]]; then
            log_warning "Memory usage: ${memory_usage_percent}%"
        else
            log_success "Memory usage: ${memory_usage_percent}%"
        fi
    else
        log_warning "Could not determine memory usage"
    fi

    return 0
}

# Check disk space
check_disk_space() {
    local disk_usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')

    if [[ -n "$disk_usage" ]]; then
        if [[ $disk_usage -gt 90 ]]; then
            log_error "High disk usage: ${disk_usage}%"
            return 1
        elif [[ $disk_usage -gt 80 ]]; then
            log_warning "Disk usage: ${disk_usage}%"
        else
            log_success "Disk usage: ${disk_usage}%"
        fi
    else
        log_warning "Could not determine disk usage"
    fi

    return 0
}

# Check process health
check_process_health() {
    local python_processes=$(pgrep -c python || echo "0")
    local uvicorn_processes=$(pgrep -c uvicorn || echo "0")

    if [[ $python_processes -eq 0 ]]; then
        log_error "No Python processes found"
        return 1
    fi

    if [[ $uvicorn_processes -eq 0 ]]; then
        log_error "No Uvicorn processes found"
        return 1
    fi

    log_success "Process health check passed (Python: $python_processes, Uvicorn: $uvicorn_processes)"
    return 0
}

# Check file system permissions
check_file_permissions() {
    local dirs_to_check=("/app/data" "/app/logs" "/app/tmp")

    for dir in "${dirs_to_check[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_error "Directory not found: $dir"
            return 1
        fi

        if [[ ! -w "$dir" ]]; then
            log_error "Directory not writable: $dir"
            return 1
        fi
    done

    log_success "File permissions check passed"
    return 0
}

# Check network connectivity
check_network_connectivity() {
    # Check if we can reach external services (if configured)
    if command -v curl &> /dev/null; then
        # Check if we can reach the metrics endpoint
        local metrics_url="http://${API_HOST}:${API_PORT}/metrics"
        if curl -f -s --max-time 5 "$metrics_url" > /dev/null 2>&1; then
            log_success "Metrics endpoint reachable"
        else
            log_warning "Metrics endpoint not reachable"
        fi
    fi

    return 0
}

# Check environment variables
check_environment() {
    local required_vars=(
        "PYNOMALY_ENV"
        "PYNOMALY_API_HOST"
        "PYNOMALY_API_PORT"
    )

    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required environment variable $var is not set"
            return 1
        fi
    done

    # Check if environment is staging
    if [[ "$PYNOMALY_ENV" != "staging" ]]; then
        log_error "Environment should be 'staging', got: $PYNOMALY_ENV"
        return 1
    fi

    log_success "Environment variables check passed"
    return 0
}

# Comprehensive health check
comprehensive_health_check() {
    log_info "Starting comprehensive health check..."

    local checks=(
        "check_environment"
        "check_application_health"
        "check_database_health"
        "check_memory_usage"
        "check_disk_space"
        "check_process_health"
        "check_file_permissions"
        "check_network_connectivity"
    )

    local failed_checks=0

    for check in "${checks[@]}"; do
        if ! $check; then
            failed_checks=$((failed_checks + 1))
        fi
    done

    if [[ $failed_checks -eq 0 ]]; then
        log_success "All health checks passed"
        return 0
    else
        log_error "$failed_checks health check(s) failed"
        return 1
    fi
}

# Simple health check (default)
simple_health_check() {
    log_info "Starting simple health check..."

    if check_application_health; then
        log_success "Simple health check passed"
        return 0
    else
        log_error "Simple health check failed"
        return 1
    fi
}

# Main function
main() {
    case "${1:-simple}" in
        "simple")
            simple_health_check
            ;;
        "comprehensive")
            comprehensive_health_check
            ;;
        "database")
            check_database_health
            ;;
        "resources")
            check_memory_usage && check_disk_space
            ;;
        *)
            log_error "Unknown health check type: $1"
            echo "Usage: $0 [simple|comprehensive|database|resources]"
            exit 1
            ;;
    esac
}

# Run main function with arguments
main "$@"

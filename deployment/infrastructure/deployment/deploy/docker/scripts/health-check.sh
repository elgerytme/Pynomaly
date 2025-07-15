#!/bin/bash

# Health Check Script for Pynomaly Production Environment
# This script performs comprehensive health checks on all services

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.production.yml"
ENV_FILE="$PROJECT_ROOT/.env.production"

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

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Health check results
declare -A health_results
overall_health=true

# Check service health
check_service_health() {
    local service_name="$1"
    local health_url="$2"
    local timeout="${3:-10}"

    log_info "Checking $service_name health..."

    if curl -f -s --max-time "$timeout" "$health_url" > /dev/null; then
        log_success "$service_name is healthy"
        health_results["$service_name"]="healthy"
        return 0
    else
        log_error "$service_name is unhealthy"
        health_results["$service_name"]="unhealthy"
        overall_health=false
        return 1
    fi
}

# Check Docker container status
check_container_status() {
    local container_name="$1"

    log_info "Checking container $container_name status..."

    local status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "not_found")

    case "$status" in
        "running")
            log_success "$container_name is running"
            health_results["$container_name"]="running"
            return 0
            ;;
        "not_found")
            log_error "$container_name not found"
            health_results["$container_name"]="not_found"
            overall_health=false
            return 1
            ;;
        *)
            log_error "$container_name status: $status"
            health_results["$container_name"]="$status"
            overall_health=false
            return 1
            ;;
    esac
}

# Check database connectivity
check_database() {
    log_info "Checking database connectivity..."

    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T postgres pg_isready -U pynomaly -d pynomaly_prod > /dev/null 2>&1; then
        log_success "Database is accessible"
        health_results["database"]="accessible"
        return 0
    else
        log_error "Database is not accessible"
        health_results["database"]="not_accessible"
        overall_health=false
        return 1
    fi
}

# Check Redis connectivity
check_redis() {
    log_info "Checking Redis connectivity..."

    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T redis-cluster redis-cli ping | grep -q "PONG"; then
        log_success "Redis is accessible"
        health_results["redis"]="accessible"
        return 0
    else
        log_error "Redis is not accessible"
        health_results["redis"]="not_accessible"
        overall_health=false
        return 1
    fi
}

# Check disk space
check_disk_space() {
    log_info "Checking disk space..."

    local data_dir="${DATA_PATH:-/opt/pynomaly/data}"
    local threshold=80

    if [[ -d "$data_dir" ]]; then
        local usage=$(df "$data_dir" | tail -1 | awk '{print $5}' | sed 's/%//')

        if [[ $usage -lt $threshold ]]; then
            log_success "Disk space usage: ${usage}% (under ${threshold}%)"
            health_results["disk_space"]="ok"
            return 0
        else
            log_error "Disk space usage: ${usage}% (over ${threshold}%)"
            health_results["disk_space"]="high"
            overall_health=false
            return 1
        fi
    else
        log_warn "Data directory not found: $data_dir"
        health_results["disk_space"]="unknown"
        return 1
    fi
}

# Check memory usage
check_memory_usage() {
    log_info "Checking memory usage..."

    local threshold=80
    local usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')

    if [[ $usage -lt $threshold ]]; then
        log_success "Memory usage: ${usage}% (under ${threshold}%)"
        health_results["memory"]="ok"
        return 0
    else
        log_error "Memory usage: ${usage}% (over ${threshold}%)"
        health_results["memory"]="high"
        overall_health=false
        return 1
    fi
}

# Check API endpoints
check_api_endpoints() {
    log_info "Checking API endpoints..."

    local base_url="http://localhost:8000"
    local endpoints=(
        "/api/health/ready"
        "/api/health/live"
        "/api/auth/health"
        "/api/datasets/health"
        "/api/detectors/health"
    )

    for endpoint in "${endpoints[@]}"; do
        if curl -f -s --max-time 5 "$base_url$endpoint" > /dev/null; then
            log_success "Endpoint $endpoint is accessible"
        else
            log_error "Endpoint $endpoint is not accessible"
            health_results["api_endpoints"]="some_failing"
            overall_health=false
        fi
    done

    if [[ "${health_results[api_endpoints]:-}" != "some_failing" ]]; then
        health_results["api_endpoints"]="all_accessible"
    fi
}

# Check log files
check_log_files() {
    log_info "Checking log files..."

    local log_dir="${LOG_PATH:-/opt/pynomaly/logs}"

    if [[ -d "$log_dir" ]]; then
        local log_files=$(find "$log_dir" -name "*.log" -mtime -1 | wc -l)

        if [[ $log_files -gt 0 ]]; then
            log_success "Found $log_files recent log files"
            health_results["log_files"]="recent_logs_found"
            return 0
        else
            log_warn "No recent log files found"
            health_results["log_files"]="no_recent_logs"
            return 1
        fi
    else
        log_warn "Log directory not found: $log_dir"
        health_results["log_files"]="directory_not_found"
        return 1
    fi
}

# Generate health report
generate_health_report() {
    log_info "Generating health report..."

    local report_file="/tmp/pynomaly_health_report_$(date +%Y%m%d_%H%M%S).json"

    cat > "$report_file" << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "overall_health": $overall_health,
    "checks": {
EOF

    local first=true
    for check in "${!health_results[@]}"; do
        if [[ "$first" == true ]]; then
            first=false
        else
            echo "," >> "$report_file"
        fi
        echo "        \"$check\": \"${health_results[$check]}\"" >> "$report_file"
    done

    cat >> "$report_file" << EOF
    }
}
EOF

    log_success "Health report generated: $report_file"

    # Display summary
    echo -e "\n${BLUE}=== HEALTH CHECK SUMMARY ===${NC}"
    if [[ "$overall_health" == true ]]; then
        echo -e "${GREEN}Overall Status: HEALTHY${NC}"
    else
        echo -e "${RED}Overall Status: UNHEALTHY${NC}"
    fi

    echo -e "\n${BLUE}Individual Checks:${NC}"
    for check in "${!health_results[@]}"; do
        local status="${health_results[$check]}"
        case "$status" in
            "healthy"|"running"|"accessible"|"ok"|"all_accessible"|"recent_logs_found")
                echo -e "  ${GREEN}✓${NC} $check: $status"
                ;;
            *)
                echo -e "  ${RED}✗${NC} $check: $status"
                ;;
        esac
    done

    echo -e "\nFull report: $report_file"
}

# Main health check function
main() {
    log_info "Starting Pynomaly production health check..."

    # Source environment variables
    if [[ -f "$ENV_FILE" ]]; then
        source "$ENV_FILE"
    fi

    # Perform health checks
    check_container_status "pynomaly-api"
    check_container_status "pynomaly-postgres"
    check_container_status "pynomaly-redis"
    check_container_status "pynomaly-prometheus"
    check_container_status "pynomaly-grafana"

    check_service_health "API Health" "http://localhost:8000/api/health/ready"
    check_service_health "Prometheus" "http://localhost:9090/-/healthy"
    check_service_health "Grafana" "http://localhost:3000/api/health"

    check_database
    check_redis
    check_disk_space
    check_memory_usage
    check_api_endpoints
    check_log_files

    # Generate report
    generate_health_report

    # Exit with appropriate code
    if [[ "$overall_health" == true ]]; then
        log_success "Health check completed successfully"
        exit 0
    else
        log_error "Health check failed"
        exit 1
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

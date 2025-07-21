#!/bin/bash

# Production Deployment Verification Script
# Comprehensive verification checklist for anomaly_detection production deployment

set -e

# Configuration
DOMAIN=${1:-localhost}
PROTOCOL=${2:-http}
API_BASE_URL="${PROTOCOL}://${DOMAIN}"
GRAFANA_URL="${PROTOCOL}://${DOMAIN}:3000"
TIMEOUT=30
LOG_FILE="/tmp/deployment_verification_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        INFO)
            echo -e "${BLUE}[INFO]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        WARNING)
            echo -e "${YELLOW}[WARNING]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Helper function to check HTTP endpoint
check_endpoint() {
    local url=$1
    local expected_status=${2:-200}
    local description=$3

    log INFO "Checking: $description"
    log INFO "URL: $url"

    if curl -f -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$url" | grep -q "$expected_status"; then
        log SUCCESS "$description is working"
        return 0
    else
        log ERROR "$description failed (URL: $url)"
        return 1
    fi
}

# Helper function to check Docker service
check_docker_service() {
    local service_name=$1
    local description=$2

    log INFO "Checking Docker service: $description"

    if docker-compose -f docker-compose.production.yml ps --services --filter "status=running" | grep -q "$service_name"; then
        local status=$(docker-compose -f docker-compose.production.yml ps "$service_name" | grep "$service_name")
        if echo "$status" | grep -q "Up"; then
            log SUCCESS "$description is running"
            return 0
        else
            log ERROR "$description is not running properly"
            log ERROR "Status: $status"
            return 1
        fi
    else
        log ERROR "$description service not found or not running"
        return 1
    fi
}

# Main verification function
main() {
    log INFO "=== Production Deployment Verification Started ==="
    log INFO "Domain: $DOMAIN"
    log INFO "Protocol: $PROTOCOL"
    log INFO "Timestamp: $(date)"
    log INFO "Log file: $LOG_FILE"

    local total_checks=0
    local passed_checks=0
    local failed_checks=0

    echo ""
    log INFO "=== 1. INFRASTRUCTURE VERIFICATION ==="

    # Check Docker services
    total_checks=$((total_checks + 6))
    check_docker_service "postgres" "PostgreSQL Database" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_docker_service "redis" "Redis Cache" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_docker_service "api" "anomaly detection API Service" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_docker_service "nginx" "Nginx Load Balancer" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_docker_service "prometheus" "Prometheus Monitoring" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_docker_service "grafana" "Grafana Dashboard" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))

    echo ""
    log INFO "=== 2. CORE API VERIFICATION ==="

    # API Health Checks
    total_checks=$((total_checks + 8))
    check_endpoint "$API_BASE_URL/health" 200 "Main Health Check" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/api/v1/health" 200 "API Health Check" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/docs" 200 "API Documentation" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/openapi.json" 200 "OpenAPI Schema" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/api/v1/datasets" 200 "Datasets Endpoint" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/api/v1/detectors" 200 "Detectors Endpoint" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/api/v1/experiments" 200 "Experiments Endpoint" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/metrics" 200 "Prometheus Metrics" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))

    echo ""
    log INFO "=== 3. WEB INTERFACE VERIFICATION ==="

    # Web Interface Checks
    total_checks=$((total_checks + 5))
    check_endpoint "$API_BASE_URL/" 200 "Main Dashboard" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/datasets" 200 "Datasets Page" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/detectors" 200 "Detectors Page" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/visualizations" 200 "Visualizations Page" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/monitoring" 200 "Monitoring Page" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))

    echo ""
    log INFO "=== 4. REAL-TIME FEATURES VERIFICATION ==="

    # Real-time and Streaming Checks
    total_checks=$((total_checks + 3))
    check_endpoint "$API_BASE_URL/api/v1/streaming/health" 200 "Streaming Health" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$API_BASE_URL/api/v1/websocket/health" 200 "WebSocket Health" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))

    # Test WebSocket connection
    log INFO "Testing WebSocket connection"
    if command -v wscat >/dev/null 2>&1; then
        if timeout 10 wscat -c "ws://${DOMAIN}/ws/streaming" -x "ping" >/dev/null 2>&1; then
            log SUCCESS "WebSocket connection test successful"
            passed_checks=$((passed_checks + 1))
        else
            log WARNING "WebSocket connection test failed (may be expected if authentication required)"
            failed_checks=$((failed_checks + 1))
        fi
    else
        log WARNING "wscat not available, skipping WebSocket test"
        failed_checks=$((failed_checks + 1))
    fi

    echo ""
    log INFO "=== 5. DATABASE VERIFICATION ==="

    # Database Checks
    total_checks=$((total_checks + 3))

    log INFO "Testing database connectivity"
    if docker-compose -f docker-compose.production.yml exec -T postgres pg_isready -U anomaly_detection -d anomaly_detection_prod >/dev/null 2>&1; then
        log SUCCESS "Database connectivity test passed"
        passed_checks=$((passed_checks + 1))
    else
        log ERROR "Database connectivity test failed"
        failed_checks=$((failed_checks + 1))
    fi

    log INFO "Checking database tables"
    if docker-compose -f docker-compose.production.yml exec -T postgres psql -U anomaly_detection -d anomaly_detection_prod -c "\dt" >/dev/null 2>&1; then
        log SUCCESS "Database tables accessible"
        passed_checks=$((passed_checks + 1))
    else
        log ERROR "Database tables not accessible"
        failed_checks=$((failed_checks + 1))
    fi

    log INFO "Testing database performance"
    if docker-compose -f docker-compose.production.yml exec -T postgres psql -U anomaly_detection -d anomaly_detection_prod -c "SELECT 1;" >/dev/null 2>&1; then
        log SUCCESS "Database query test passed"
        passed_checks=$((passed_checks + 1))
    else
        log ERROR "Database query test failed"
        failed_checks=$((failed_checks + 1))
    fi

    echo ""
    log INFO "=== 6. CACHE VERIFICATION ==="

    # Redis Cache Checks
    total_checks=$((total_checks + 2))

    log INFO "Testing Redis connectivity"
    if docker-compose -f docker-compose.production.yml exec -T redis redis-cli ping >/dev/null 2>&1; then
        log SUCCESS "Redis connectivity test passed"
        passed_checks=$((passed_checks + 1))
    else
        log ERROR "Redis connectivity test failed"
        failed_checks=$((failed_checks + 1))
    fi

    log INFO "Testing Redis operations"
    if docker-compose -f docker-compose.production.yml exec -T redis redis-cli set test_key "test_value" >/dev/null 2>&1 && \
       docker-compose -f docker-compose.production.yml exec -T redis redis-cli get test_key >/dev/null 2>&1 && \
       docker-compose -f docker-compose.production.yml exec -T redis redis-cli del test_key >/dev/null 2>&1; then
        log SUCCESS "Redis operations test passed"
        passed_checks=$((passed_checks + 1))
    else
        log ERROR "Redis operations test failed"
        failed_checks=$((failed_checks + 1))
    fi

    echo ""
    log INFO "=== 7. MONITORING VERIFICATION ==="

    # Monitoring Checks
    total_checks=$((total_checks + 3))
    check_endpoint "http://${DOMAIN}:9090/api/v1/query?query=up" 200 "Prometheus API" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "$GRAFANA_URL/api/health" 200 "Grafana API" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))
    check_endpoint "http://${DOMAIN}:9093/api/v1/status" 200 "AlertManager API" && passed_checks=$((passed_checks + 1)) || failed_checks=$((failed_checks + 1))

    echo ""
    log INFO "=== 8. SECURITY VERIFICATION ==="

    # Security Checks
    total_checks=$((total_checks + 4))

    log INFO "Checking SSL certificate (if HTTPS)"
    if [ "$PROTOCOL" = "https" ]; then
        if openssl s_client -connect "${DOMAIN}:443" -servername "$DOMAIN" </dev/null 2>/dev/null | openssl x509 -noout -dates >/dev/null 2>&1; then
            log SUCCESS "SSL certificate validation passed"
            passed_checks=$((passed_checks + 1))
        else
            log ERROR "SSL certificate validation failed"
            failed_checks=$((failed_checks + 1))
        fi
    else
        log WARNING "HTTP mode - SSL certificate check skipped"
        failed_checks=$((failed_checks + 1))
    fi

    log INFO "Checking security headers"
    if curl -I -s "$API_BASE_URL" | grep -i "x-frame-options\|x-content-type-options\|strict-transport-security" >/dev/null; then
        log SUCCESS "Security headers present"
        passed_checks=$((passed_checks + 1))
    else
        log WARNING "Some security headers missing"
        failed_checks=$((failed_checks + 1))
    fi

    log INFO "Testing rate limiting"
    local rate_limit_test=0
    for i in {1..20}; do
        if curl -f -s -o /dev/null "$API_BASE_URL/health" 2>/dev/null; then
            rate_limit_test=$((rate_limit_test + 1))
        fi
    done

    if [ $rate_limit_test -eq 20 ]; then
        log WARNING "Rate limiting may not be configured properly (all 20 requests succeeded)"
        failed_checks=$((failed_checks + 1))
    else
        log SUCCESS "Rate limiting appears to be working"
        passed_checks=$((passed_checks + 1))
    fi

    log INFO "Checking for exposed sensitive endpoints"
    if curl -f -s -o /dev/null "$API_BASE_URL/admin" 2>/dev/null || \
       curl -f -s -o /dev/null "$API_BASE_URL/.env" 2>/dev/null || \
       curl -f -s -o /dev/null "$API_BASE_URL/config" 2>/dev/null; then
        log ERROR "Sensitive endpoints may be exposed"
        failed_checks=$((failed_checks + 1))
    else
        log SUCCESS "No sensitive endpoints exposed"
        passed_checks=$((passed_checks + 1))
    fi

    echo ""
    log INFO "=== 9. PERFORMANCE VERIFICATION ==="

    # Performance Checks
    total_checks=$((total_checks + 3))

    log INFO "Testing API response time"
    local response_time=$(curl -w "%{time_total}" -o /dev/null -s "$API_BASE_URL/health")
    if (( $(echo "$response_time < 1.0" | bc -l) )); then
        log SUCCESS "API response time acceptable: ${response_time}s"
        passed_checks=$((passed_checks + 1))
    else
        log WARNING "API response time slow: ${response_time}s"
        failed_checks=$((failed_checks + 1))
    fi

    log INFO "Checking container resource usage"
    local high_memory_containers=$(docker stats --no-stream --format "table {{.Container}}\t{{.MemPerc}}" | awk 'NR>1 && $2+0 > 80 {print $1}')
    if [ -z "$high_memory_containers" ]; then
        log SUCCESS "All containers within memory limits"
        passed_checks=$((passed_checks + 1))
    else
        log WARNING "High memory usage detected in: $high_memory_containers"
        failed_checks=$((failed_checks + 1))
    fi

    log INFO "Checking disk space"
    local disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 80 ]; then
        log SUCCESS "Disk space usage acceptable: ${disk_usage}%"
        passed_checks=$((passed_checks + 1))
    else
        log WARNING "Disk space usage high: ${disk_usage}%"
        failed_checks=$((failed_checks + 1))
    fi

    echo ""
    log INFO "=== 10. BACKUP VERIFICATION ==="

    # Backup Checks
    total_checks=$((total_checks + 2))

    log INFO "Checking backup directory"
    if [ -d "./backups" ] && [ "$(ls -A ./backups 2>/dev/null)" ]; then
        log SUCCESS "Backup directory exists and contains files"
        passed_checks=$((passed_checks + 1))
    else
        log WARNING "Backup directory empty or missing"
        failed_checks=$((failed_checks + 1))
    fi

    log INFO "Testing backup script"
    if [ -f "./scripts/backup_database.sh" ] && [ -x "./scripts/backup_database.sh" ]; then
        log SUCCESS "Backup script exists and is executable"
        passed_checks=$((passed_checks + 1))
    else
        log WARNING "Backup script missing or not executable"
        failed_checks=$((failed_checks + 1))
    fi

    echo ""
    log INFO "=== VERIFICATION SUMMARY ==="
    log INFO "Total checks: $total_checks"
    log SUCCESS "Passed: $passed_checks"
    log ERROR "Failed: $failed_checks"

    local success_rate=$(( passed_checks * 100 / total_checks ))
    log INFO "Success rate: ${success_rate}%"

    if [ $success_rate -ge 90 ]; then
        log SUCCESS "=== DEPLOYMENT VERIFICATION PASSED ==="
        log SUCCESS "Production deployment is ready for use!"
    elif [ $success_rate -ge 70 ]; then
        log WARNING "=== DEPLOYMENT VERIFICATION PARTIAL ==="
        log WARNING "Production deployment has some issues but is mostly functional"
    else
        log ERROR "=== DEPLOYMENT VERIFICATION FAILED ==="
        log ERROR "Production deployment has significant issues and needs attention"
    fi

    echo ""
    log INFO "Detailed verification log saved to: $LOG_FILE"
    log INFO "=== Production Deployment Verification Completed ==="

    # Exit with appropriate code
    if [ $success_rate -ge 70 ]; then
        exit 0
    else
        exit 1
    fi
}

# Script usage
usage() {
    echo "Usage: $0 [domain] [protocol]"
    echo "  domain   - Domain name or IP address (default: localhost)"
    echo "  protocol - http or https (default: http)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Verify localhost with HTTP"
    echo "  $0 my-domain.com https      # Verify production domain with HTTPS"
    echo "  $0 192.168.1.100 http      # Verify specific IP with HTTP"
    exit 1
}

# Check if help requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

# Check dependencies
log INFO "Checking script dependencies..."

if ! command -v curl >/dev/null 2>&1; then
    log ERROR "curl is required but not installed"
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    log ERROR "docker is required but not installed"
    exit 1
fi

if ! command -v docker-compose >/dev/null 2>&1; then
    log ERROR "docker-compose is required but not installed"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "docker-compose.production.yml" ]; then
    log ERROR "docker-compose.production.yml not found. Please run this script from the anomaly_detection root directory."
    exit 1
fi

# Run main verification
main

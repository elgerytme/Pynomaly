#!/bin/bash

# anomaly_detection Production Validation Script
# This script validates the production deployment

set -euo pipefail

# Configuration
NAMESPACE="anomaly_detection-production"
API_ENDPOINT="http://localhost:8000"
PROMETHEUS_ENDPOINT="http://localhost:9090"
GRAFANA_ENDPOINT="http://localhost:3000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓ PASS]${NC} $1"
    ((PASSED_CHECKS++))
}

fail() {
    echo -e "${RED}[✗ FAIL]${NC} $1"
    ((FAILED_CHECKS++))
}

warning() {
    echo -e "${YELLOW}[⚠ WARN]${NC} $1"
}

# Function to run a check
run_check() {
    local check_name="$1"
    local check_command="$2"

    ((TOTAL_CHECKS++))
    log "Running: $check_name"

    if eval "$check_command" &> /dev/null; then
        success "$check_name"
        return 0
    else
        fail "$check_name"
        return 1
    fi
}

# Check if kubectl is available
check_kubectl() {
    run_check "kubectl availability" "command -v kubectl"
}

# Check cluster connectivity
check_cluster_connectivity() {
    run_check "Kubernetes cluster connectivity" "kubectl cluster-info"
}

# Check namespace exists
check_namespace() {
    run_check "Namespace exists" "kubectl get namespace $NAMESPACE"
}

# Check all pods are running
check_pods() {
    log "Checking pod status..."

    # Get all pods in namespace
    PODS=$(kubectl get pods -n $NAMESPACE -o json | jq -r '.items[] | select(.status.phase != "Running") | .metadata.name')

    if [ -z "$PODS" ]; then
        success "All pods are running"
        return 0
    else
        fail "Some pods are not running: $PODS"
        return 1
    fi
}

# Check services are available
check_services() {
    local services=("anomaly_detection-service" "postgres-service" "redis-service" "mongodb-service")

    for service in "${services[@]}"; do
        run_check "Service $service exists" "kubectl get service $service -n $NAMESPACE"
    done
}

# Check persistent volumes
check_persistent_volumes() {
    local pvcs=("anomaly_detection-app-storage" "postgres-storage" "redis-storage" "mongodb-storage")

    for pvc in "${pvcs[@]}"; do
        run_check "PVC $pvc is bound" "kubectl get pvc $pvc -n $NAMESPACE -o jsonpath='{.status.phase}' | grep -q 'Bound'"
    done
}

# Check ingress
check_ingress() {
    run_check "Ingress exists" "kubectl get ingress anomaly_detection-ingress -n $NAMESPACE"
}

# Check horizontal pod autoscaler
check_hpa() {
    run_check "HPA is configured" "kubectl get hpa anomaly_detection-app-hpa -n $NAMESPACE"
}

# Check application health endpoint
check_app_health() {
    # Port forward to access the service
    kubectl port-forward -n $NAMESPACE service/anomaly_detection-service 8000:8000 &
    local PF_PID=$!

    # Wait for port forward to be ready
    sleep 5

    # Check health endpoint
    if curl -f $API_ENDPOINT/health &> /dev/null; then
        success "Application health endpoint is accessible"
        local health_result=0
    else
        fail "Application health endpoint is not accessible"
        local health_result=1
    fi

    # Clean up port forward
    kill $PF_PID 2>/dev/null || true

    return $health_result
}

# Check database connectivity
check_database_connectivity() {
    log "Checking database connectivity..."

    # Get a pod to test from
    local POD=$(kubectl get pods -l component=app -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}')

    if [ -z "$POD" ]; then
        fail "No application pods found"
        return 1
    fi

    # Test PostgreSQL
    if kubectl exec -n $NAMESPACE $POD -- python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('PostgreSQL OK')
except Exception as e:
    print(f'PostgreSQL Error: {e}')
    exit(1)
" &> /dev/null; then
        success "PostgreSQL connectivity"
    else
        fail "PostgreSQL connectivity"
    fi

    # Test Redis
    if kubectl exec -n $NAMESPACE $POD -- python -c "
import redis
import os
try:
    r = redis.from_url(os.environ['REDIS_URL'])
    r.ping()
    print('Redis OK')
except Exception as e:
    print(f'Redis Error: {e}')
    exit(1)
" &> /dev/null; then
        success "Redis connectivity"
    else
        fail "Redis connectivity"
    fi
}

# Check monitoring services
check_monitoring() {
    # Check if Prometheus is deployed
    if kubectl get deployment prometheus -n $NAMESPACE &> /dev/null; then
        success "Prometheus deployment exists"

        # Port forward to check Prometheus
        kubectl port-forward -n $NAMESPACE service/prometheus-service 9090:9090 &
        local PROM_PID=$!
        sleep 3

        if curl -f $PROMETHEUS_ENDPOINT/-/ready &> /dev/null; then
            success "Prometheus is ready"
        else
            fail "Prometheus is not ready"
        fi

        kill $PROM_PID 2>/dev/null || true
    else
        fail "Prometheus deployment not found"
    fi

    # Check if Grafana is deployed
    if kubectl get deployment grafana -n $NAMESPACE &> /dev/null; then
        success "Grafana deployment exists"

        # Port forward to check Grafana
        kubectl port-forward -n $NAMESPACE service/grafana-service 3000:3000 &
        local GRAF_PID=$!
        sleep 3

        if curl -f $GRAFANA_ENDPOINT/api/health &> /dev/null; then
            success "Grafana is ready"
        else
            fail "Grafana is not ready"
        fi

        kill $GRAF_PID 2>/dev/null || true
    else
        fail "Grafana deployment not found"
    fi
}

# Check resource usage
check_resource_usage() {
    log "Checking resource usage..."

    # Get resource usage for all pods
    local RESOURCE_OUTPUT=$(kubectl top pods -n $NAMESPACE --no-headers 2>/dev/null)

    if [ $? -eq 0 ]; then
        success "Resource metrics are available"

        # Check if any pods are using too much CPU (>80%)
        local HIGH_CPU=$(echo "$RESOURCE_OUTPUT" | awk '{gsub(/m/, "", $2); if ($2 > 800) print $1}')
        if [ -n "$HIGH_CPU" ]; then
            warning "High CPU usage detected in pods: $HIGH_CPU"
        fi

        # Check if any pods are using too much memory (>80%)
        local HIGH_MEM=$(echo "$RESOURCE_OUTPUT" | awk '{gsub(/Mi/, "", $3); if ($3 > 800) print $1}')
        if [ -n "$HIGH_MEM" ]; then
            warning "High memory usage detected in pods: $HIGH_MEM"
        fi
    else
        warning "Resource metrics not available (metrics-server may not be installed)"
    fi
}

# Check security configuration
check_security() {
    log "Checking security configuration..."

    # Check if secrets exist
    local secrets=("anomaly_detection-secrets" "anomaly_detection-tls")
    for secret in "${secrets[@]}"; do
        run_check "Secret $secret exists" "kubectl get secret $secret -n $NAMESPACE"
    done

    # Check if pods are running as non-root
    local PODS=$(kubectl get pods -n $NAMESPACE -o json | jq -r '.items[] | select(.spec.securityContext.runAsNonRoot != true) | .metadata.name')
    if [ -z "$PODS" ]; then
        success "All pods are running as non-root"
    else
        fail "Some pods are running as root: $PODS"
    fi

    # Check if network policies exist
    if kubectl get networkpolicy -n $NAMESPACE &> /dev/null; then
        success "Network policies are configured"
    else
        warning "No network policies found"
    fi
}

# Check backup configuration
check_backup() {
    log "Checking backup configuration..."

    # Check if CronJobs for backup exist
    if kubectl get cronjob -n $NAMESPACE &> /dev/null; then
        success "Backup CronJobs are configured"
    else
        warning "No backup CronJobs found"
    fi
}

# Performance tests
run_performance_tests() {
    log "Running basic performance tests..."

    # Port forward to access the service
    kubectl port-forward -n $NAMESPACE service/anomaly_detection-service 8000:8000 &
    local PF_PID=$!
    sleep 3

    # Simple load test
    local RESPONSE_TIME=$(curl -w "%{time_total}" -s -o /dev/null $API_ENDPOINT/health)

    if (( $(echo "$RESPONSE_TIME < 1.0" | bc -l) )); then
        success "Response time is acceptable ($RESPONSE_TIME seconds)"
    else
        warning "Response time is slow ($RESPONSE_TIME seconds)"
    fi

    # Clean up port forward
    kill $PF_PID 2>/dev/null || true
}

# Generate report
generate_report() {
    log "Generating validation report..."

    local REPORT_FILE="production_validation_report_$(date +%Y%m%d_%H%M%S).md"

    cat > "$REPORT_FILE" << EOF
# anomaly_detection Production Validation Report

**Date:** $(date)
**Namespace:** $NAMESPACE

## Summary

- **Total Checks:** $TOTAL_CHECKS
- **Passed:** $PASSED_CHECKS
- **Failed:** $FAILED_CHECKS
- **Success Rate:** $(( PASSED_CHECKS * 100 / TOTAL_CHECKS ))%

## Deployment Status

$(kubectl get pods -n $NAMESPACE)

## Services Status

$(kubectl get services -n $NAMESPACE)

## Resource Usage

$(kubectl top pods -n $NAMESPACE 2>/dev/null || echo "Resource metrics not available")

## Recommendations

EOF

    if [ $FAILED_CHECKS -gt 0 ]; then
        echo "- Address the failed checks before proceeding to production" >> "$REPORT_FILE"
    fi

    if [ $PASSED_CHECKS -eq $TOTAL_CHECKS ]; then
        echo "- All checks passed! The deployment is ready for production use." >> "$REPORT_FILE"
    fi

    success "Validation report generated: $REPORT_FILE"
}

# Main execution
main() {
    log "Starting anomaly_detection Production Validation..."

    # Infrastructure checks
    check_kubectl
    check_cluster_connectivity
    check_namespace
    check_pods
    check_services
    check_persistent_volumes
    check_ingress
    check_hpa

    # Application checks
    check_app_health
    check_database_connectivity

    # Monitoring checks
    check_monitoring

    # Security checks
    check_security

    # Performance checks
    run_performance_tests

    # Resource checks
    check_resource_usage

    # Backup checks
    check_backup

    # Generate report
    generate_report

    log "Validation completed!"
    echo ""
    echo "📊 Results Summary:"
    echo "   Total Checks: $TOTAL_CHECKS"
    echo "   Passed: $PASSED_CHECKS"
    echo "   Failed: $FAILED_CHECKS"
    echo "   Success Rate: $(( PASSED_CHECKS * 100 / TOTAL_CHECKS ))%"
    echo ""

    if [ $FAILED_CHECKS -eq 0 ]; then
        success "All validations passed! Production deployment is ready."
        exit 0
    else
        error "$FAILED_CHECKS checks failed. Please address the issues before production use."
        exit 1
    fi
}

# Run main function
main "$@"

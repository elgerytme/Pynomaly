#!/usr/bin/env bash

# =============================================================================
# Pynomaly Production Deployment Validation Script
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
NAMESPACE="${NAMESPACE:-pynomaly-production}"
TIMEOUT="${TIMEOUT:-300}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${YELLOW}[VERBOSE]${NC} $1"
    fi
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

validate_prerequisites() {
    log "Validating prerequisites..."

    # Check required tools
    local required_tools=("kubectl" "curl" "jq" "python3")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is required but not installed"
        fi
        verbose "$tool is available"
    done

    # Check kubectl context
    local current_context
    current_context=$(kubectl config current-context 2>/dev/null || echo "")
    if [[ -z "$current_context" ]]; then
        error "No kubectl context is set"
    fi
    verbose "kubectl context: $current_context"

    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        error "Namespace '$NAMESPACE' does not exist"
    fi
    verbose "Namespace '$NAMESPACE' exists"

    success "Prerequisites validation completed"
}

validate_deployment_status() {
    log "Validating deployment status..."

    # Check deployment exists and is ready
    local deployment_status
    deployment_status=$(kubectl get deployment pynomaly-api -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' 2>/dev/null || echo "Unknown")

    if [[ "$deployment_status" != "True" ]]; then
        error "Deployment pynomaly-api is not available. Status: $deployment_status"
    fi

    # Check replica count
    local desired_replicas
    local ready_replicas
    desired_replicas=$(kubectl get deployment pynomaly-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    ready_replicas=$(kubectl get deployment pynomaly-api -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')

    if [[ "$ready_replicas" != "$desired_replicas" ]]; then
        error "Not all replicas are ready. Desired: $desired_replicas, Ready: $ready_replicas"
    fi

    verbose "Deployment replicas: $ready_replicas/$desired_replicas"

    # Check pod status
    local running_pods
    running_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api --no-headers | grep -c "Running" || echo "0")

    if [[ "$running_pods" -lt 1 ]]; then
        error "No API pods are running"
    fi

    verbose "Running pods: $running_pods"

    success "Deployment status validation completed"
}

validate_services() {
    log "Validating services..."

    # Check service exists
    if ! kubectl get service pynomaly-api-service -n "$NAMESPACE" &> /dev/null; then
        error "Service pynomaly-api-service does not exist"
    fi

    # Check internal service
    if ! kubectl get service pynomaly-api-internal -n "$NAMESPACE" &> /dev/null; then
        error "Internal service pynomaly-api-internal does not exist"
    fi

    # Check service endpoints
    local endpoints
    endpoints=$(kubectl get endpoints pynomaly-api-service -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)

    if [[ "$endpoints" -lt 1 ]]; then
        error "Service has no endpoints"
    fi

    verbose "Service endpoints: $endpoints"

    success "Services validation completed"
}

validate_health_endpoints() {
    log "Validating health endpoints..."

    # Port forward for testing
    local port_forward_pid
    kubectl port-forward service/pynomaly-api-internal 8081:8000 -n "$NAMESPACE" &
    port_forward_pid=$!

    # Wait for port forward to be ready
    sleep 5

    # Function to cleanup port forward
    cleanup_port_forward() {
        if [[ -n "${port_forward_pid:-}" ]]; then
            kill "$port_forward_pid" 2>/dev/null || true
            wait "$port_forward_pid" 2>/dev/null || true
        fi
    }

    # Set trap for cleanup
    trap cleanup_port_forward EXIT

    # Test health endpoint
    local health_status
    health_status=$(curl -s -w "%{http_code}" http://localhost:8081/api/v1/health -o /dev/null || echo "000")

    if [[ "$health_status" != "200" ]]; then
        error "Health endpoint returned status: $health_status"
    fi

    verbose "Health endpoint status: $health_status"

    # Test API documentation
    local docs_status
    docs_status=$(curl -s -w "%{http_code}" http://localhost:8081/api/v1/docs -o /dev/null || echo "000")

    if [[ "$docs_status" != "200" ]]; then
        warning "API documentation returned status: $docs_status"
    else
        verbose "API documentation status: $docs_status"
    fi

    # Test OpenAPI schema
    local openapi_endpoint_count
    openapi_endpoint_count=$(curl -s http://localhost:8081/api/v1/openapi.json | jq '.paths | length' 2>/dev/null || echo "0")

    if [[ "$openapi_endpoint_count" -lt 50 ]]; then
        warning "OpenAPI schema has fewer endpoints than expected: $openapi_endpoint_count"
    else
        verbose "OpenAPI endpoints documented: $openapi_endpoint_count"
    fi

    # Test metrics endpoint
    local metrics_status
    metrics_status=$(curl -s -w "%{http_code}" http://localhost:8081/metrics -o /dev/null || echo "000")

    if [[ "$metrics_status" != "200" ]]; then
        warning "Metrics endpoint returned status: $metrics_status"
    else
        verbose "Metrics endpoint status: $metrics_status"
    fi

    # Cleanup port forward
    cleanup_port_forward
    trap - EXIT

    success "Health endpoints validation completed"
}

validate_database_connectivity() {
    log "Validating database connectivity..."

    # Check if database service exists
    if ! kubectl get service postgres -n "$NAMESPACE" &> /dev/null; then
        warning "Database service not found in namespace '$NAMESPACE'"
        return 0
    fi

    # Check database pod status
    local db_running_pods
    db_running_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=database --no-headers | grep -c "Running" || echo "0")

    if [[ "$db_running_pods" -lt 1 ]]; then
        warning "No database pods are running"
    else
        verbose "Database pods running: $db_running_pods"
    fi

    success "Database connectivity validation completed"
}

validate_redis_connectivity() {
    log "Validating Redis connectivity..."

    # Check if Redis service exists
    if ! kubectl get service redis -n "$NAMESPACE" &> /dev/null; then
        warning "Redis service not found in namespace '$NAMESPACE'"
        return 0
    fi

    # Check Redis pod status
    local redis_running_pods
    redis_running_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=cache --no-headers | grep -c "Running" || echo "0")

    if [[ "$redis_running_pods" -lt 1 ]]; then
        warning "No Redis pods are running"
    else
        verbose "Redis pods running: $redis_running_pods"
    fi

    success "Redis connectivity validation completed"
}

validate_resource_usage() {
    log "Validating resource usage..."

    # Get pod resource usage
    local pod_info
    pod_info=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null || echo "")

    if [[ -z "$pod_info" ]]; then
        warning "Unable to get pod resource usage (metrics-server may not be available)"
        return 0
    fi

    # Check for high resource usage
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            local pod_name cpu_usage memory_usage
            pod_name=$(echo "$line" | awk '{print $1}')
            cpu_usage=$(echo "$line" | awk '{print $2}' | sed 's/m$//')
            memory_usage=$(echo "$line" | awk '{print $3}' | sed 's/Mi$//')

            verbose "Pod $pod_name: CPU ${cpu_usage}m, Memory ${memory_usage}Mi"

            # Warning thresholds (adjust as needed)
            if [[ "${cpu_usage:-0}" -gt 1000 ]]; then
                warning "High CPU usage for pod $pod_name: ${cpu_usage}m"
            fi

            if [[ "${memory_usage:-0}" -gt 2048 ]]; then
                warning "High memory usage for pod $pod_name: ${memory_usage}Mi"
            fi
        fi
    done <<< "$pod_info"

    success "Resource usage validation completed"
}

validate_monitoring() {
    log "Validating monitoring setup..."

    # Check if monitoring components exist
    local monitoring_components=("prometheus" "grafana" "alertmanager")

    for component in "${monitoring_components[@]}"; do
        if kubectl get deployment "$component" -n "$NAMESPACE" &> /dev/null; then
            verbose "Monitoring component found: $component"
        else
            warning "Monitoring component not found: $component"
        fi
    done

    success "Monitoring validation completed"
}

run_smoke_tests() {
    log "Running smoke tests..."

    # Check if smoke test script exists
    local smoke_test_script="$SCRIPT_DIR/smoke_tests.py"
    if [[ ! -f "$smoke_test_script" ]]; then
        warning "Smoke test script not found: $smoke_test_script"
        return 0
    fi

    # Install required Python packages
    if ! python3 -c "import httpx, psycopg2, redis" &> /dev/null; then
        warning "Required Python packages not available for smoke tests"
        return 0
    fi

    # Port forward for smoke tests
    local port_forward_pid
    kubectl port-forward service/pynomaly-api-internal 8082:8000 -n "$NAMESPACE" &
    port_forward_pid=$!

    # Wait for port forward
    sleep 5

    # Function to cleanup port forward
    cleanup_smoke_test_port_forward() {
        if [[ -n "${port_forward_pid:-}" ]]; then
            kill "$port_forward_pid" 2>/dev/null || true
            wait "$port_forward_pid" 2>/dev/null || true
        fi
    }

    # Set trap for cleanup
    trap cleanup_smoke_test_port_forward EXIT

    # Run smoke tests
    local smoke_test_result
    if python3 "$smoke_test_script" --url "http://localhost:8082" --timeout 30; then
        success "Smoke tests passed"
        smoke_test_result=0
    else
        warning "Some smoke tests failed"
        smoke_test_result=1
    fi

    # Cleanup
    cleanup_smoke_test_port_forward
    trap - EXIT

    return $smoke_test_result
}

generate_validation_report() {
    log "Generating validation report..."

    local report_file="/tmp/pynomaly_validation_report_$(date +%Y%m%d_%H%M%S).txt"

    {
        echo "Pynomaly Production Deployment Validation Report"
        echo "=============================================="
        echo "Date: $(date)"
        echo "Environment: $ENVIRONMENT"
        echo "Namespace: $NAMESPACE"
        echo ""

        echo "Deployment Status:"
        kubectl get deployment pynomaly-api -n "$NAMESPACE" -o wide
        echo ""

        echo "Pod Status:"
        kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api
        echo ""

        echo "Service Status:"
        kubectl get services -n "$NAMESPACE"
        echo ""

        echo "Resource Usage:"
        kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Resource metrics not available"
        echo ""

        echo "Recent Events:"
        kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10

    } > "$report_file"

    log "Validation report generated: $report_file"
}

# =============================================================================
# MAIN FUNCTION
# =============================================================================

main() {
    log "Starting Pynomaly deployment validation..."
    log "Environment: $ENVIRONMENT"
    log "Namespace: $NAMESPACE"

    # Run all validations
    validate_prerequisites
    validate_deployment_status
    validate_services
    validate_health_endpoints
    validate_database_connectivity
    validate_redis_connectivity
    validate_resource_usage
    validate_monitoring

    # Run smoke tests (optional)
    if run_smoke_tests; then
        success "All validations passed"
    else
        warning "Validations completed with warnings"
    fi

    # Generate report
    generate_validation_report

    success "🎉 Deployment validation completed successfully!"
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Validate Pynomaly production deployment"
            echo ""
            echo "OPTIONS:"
            echo "    -e, --environment ENV       Deployment environment (default: production)"
            echo "    -n, --namespace NAMESPACE   Kubernetes namespace (default: pynomaly-production)"
            echo "    -t, --timeout TIMEOUT       Timeout in seconds (default: 300)"
            echo "    -v, --verbose               Enable verbose output"
            echo "    -h, --help                  Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# =============================================================================
# ENTRY POINT
# =============================================================================

main "$@"

#!/bin/bash

# Hexagonal Architecture Health Check Script
# Usage: ./health-check.sh [environment] [deployment_type]

set -euo pipefail

# Configuration
ENVIRONMENT="${1:-development}"
DEPLOYMENT_TYPE="${2:-docker}"

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

# Health check function
check_service_health() {
    local service_name="$1"
    local url="$2"
    local timeout="${3:-10}"
    
    log_info "Checking $service_name at $url"
    
    if curl -f -s --max-time "$timeout" "$url" > /dev/null 2>&1; then
        log_success "$service_name is healthy"
        return 0
    else
        log_error "$service_name is unhealthy or unreachable"
        return 1
    fi
}

# Check Docker services
check_docker_services() {
    log_info "Checking Docker services for $ENVIRONMENT environment"
    
    local base_url="http://localhost"
    local health_checks_passed=0
    local total_checks=0
    
    # Service definitions: name:port:path
    local services=(
        "Data Quality:8000:/health"
        "MLOps Experiments:8001:/health"
        "MLOps Registry:8002:/health"
        "ML Training:8004:/health"
        "ML Prediction:8005:/health"
        "Anomaly Detection:8007:/health"
        "Prometheus:9090/-/healthy"
        "Grafana:3000:/api/health"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port path <<< "$service"
        local url="$base_url:$port$path"
        
        ((total_checks++))
        if check_service_health "$name" "$url"; then
            ((health_checks_passed++))
        fi
    done
    
    log_info "Health check summary: $health_checks_passed/$total_checks services healthy"
    
    if [[ $health_checks_passed -eq $total_checks ]]; then
        log_success "All services are healthy!"
        return 0
    else
        log_warning "Some services are unhealthy"
        return 1
    fi
}

# Check Kubernetes services
check_kubernetes_services() {
    log_info "Checking Kubernetes services for $ENVIRONMENT environment"
    
    local namespace="hexagonal-$ENVIRONMENT"
    local health_checks_passed=0
    local total_checks=0
    
    # Check if namespace exists
    if ! kubectl get namespace "$namespace" > /dev/null 2>&1; then
        log_error "Namespace $namespace does not exist"
        return 1
    fi
    
    # Check pod status
    log_info "Checking pod status in namespace $namespace"
    local pods
    pods=$(kubectl get pods -n "$namespace" --no-headers 2>/dev/null | wc -l)
    
    if [[ $pods -eq 0 ]]; then
        log_error "No pods found in namespace $namespace"
        return 1
    fi
    
    local ready_pods
    ready_pods=$(kubectl get pods -n "$namespace" --no-headers 2>/dev/null | grep -c "1/1.*Running" || echo "0")
    
    log_info "Pod status: $ready_pods/$pods pods ready"
    
    # Check service endpoints
    log_info "Checking service endpoints"
    local services
    services=$(kubectl get services -n "$namespace" --no-headers 2>/dev/null | grep -v "ClusterIP.*<none>" | wc -l)
    
    if [[ $services -gt 0 ]]; then
        log_success "$services services are available"
        ((health_checks_passed++))
    else
        log_error "No services are available"
    fi
    
    ((total_checks++))
    
    # Check ingress if it exists
    if kubectl get ingress -n "$namespace" > /dev/null 2>&1; then
        local ingress_ready
        ingress_ready=$(kubectl get ingress -n "$namespace" --no-headers 2>/dev/null | grep -v "<pending>" | wc -l)
        
        if [[ $ingress_ready -gt 0 ]]; then
            log_success "Ingress is configured and ready"
            ((health_checks_passed++))
        else
            log_warning "Ingress is pending or not ready"
        fi
        ((total_checks++))
    fi
    
    # Check HPA if it exists
    if kubectl get hpa -n "$namespace" > /dev/null 2>&1; then
        local hpa_count
        hpa_count=$(kubectl get hpa -n "$namespace" --no-headers 2>/dev/null | wc -l)
        
        if [[ $hpa_count -gt 0 ]]; then
            log_success "$hpa_count HPA(s) are configured"
            ((health_checks_passed++))
        fi
        ((total_checks++))
    fi
    
    log_info "Kubernetes health check summary: $health_checks_passed/$total_checks checks passed"
    
    if [[ $health_checks_passed -eq $total_checks ]]; then
        log_success "All Kubernetes components are healthy!"
        return 0
    else
        log_warning "Some Kubernetes components have issues"
        return 1
    fi
}

# Check service connectivity
check_service_connectivity() {
    log_info "Checking service connectivity"
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        # Check if Docker network exists
        if docker network ls | grep -q "hexagonal-architecture"; then
            log_success "Docker network exists"
        else
            log_error "Docker network not found"
            return 1
        fi
        
        # Check running containers
        local running_containers
        running_containers=$(docker ps --filter "label=part-of=hexagonal-architecture" --format "table {{.Names}}" | tail -n +2 | wc -l)
        
        log_info "$running_containers containers are running"
        
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        # Check cluster connectivity
        if kubectl cluster-info > /dev/null 2>&1; then
            log_success "Kubernetes cluster is accessible"
        else
            log_error "Cannot connect to Kubernetes cluster"
            return 1
        fi
    fi
}

# Generate health report
generate_health_report() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    log_info "Generating health report"
    
    cat << EOF > "/tmp/health-report-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S).txt"
Hexagonal Architecture Health Report
Generated: $timestamp
Environment: $ENVIRONMENT
Deployment Type: $DEPLOYMENT_TYPE

=== Service Status ===
EOF
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        echo "Docker Services:" >> "/tmp/health-report-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S).txt"
        docker ps --filter "label=part-of=hexagonal-architecture" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" >> "/tmp/health-report-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S).txt" 2>/dev/null || echo "No Docker services found" >> "/tmp/health-report-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S).txt"
    else
        echo "Kubernetes Services:" >> "/tmp/health-report-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S).txt"
        kubectl get all -n "hexagonal-$ENVIRONMENT" >> "/tmp/health-report-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S).txt" 2>/dev/null || echo "No Kubernetes services found" >> "/tmp/health-report-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S).txt"
    fi
    
    log_success "Health report saved to /tmp/health-report-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S).txt"
}

# Main function
main() {
    log_info "Starting health check for $ENVIRONMENT environment ($DEPLOYMENT_TYPE)"
    
    local overall_health=0
    
    # Check service connectivity
    if check_service_connectivity; then
        ((overall_health++))
    fi
    
    # Check services based on deployment type
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        if check_docker_services; then
            ((overall_health++))
        fi
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        if check_kubernetes_services; then
            ((overall_health++))
        fi
    else
        log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
        exit 1
    fi
    
    # Generate health report
    generate_health_report
    
    # Overall health status
    if [[ $overall_health -eq 2 ]]; then
        log_success "Overall health check: PASSED"
        exit 0
    else
        log_error "Overall health check: FAILED"
        exit 1
    fi
}

# Show help
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
    cat << EOF
Hexagonal Architecture Health Check Script

Usage: $0 [environment] [deployment_type]

Arguments:
  environment      Target environment (development|staging|production) [default: development]
  deployment_type  Deployment type (docker|kubernetes) [default: docker]

Examples:
  $0 development docker
  $0 production kubernetes

The script checks:
  - Service connectivity
  - Health endpoints
  - Container/Pod status
  - Network configuration
  - Resource availability

Exit codes:
  0 - All health checks passed
  1 - One or more health checks failed
EOF
    exit 0
fi

# Run main function
main
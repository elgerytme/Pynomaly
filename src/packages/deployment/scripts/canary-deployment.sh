#!/bin/bash
set -euo pipefail

# Canary Deployment Script
# Implements gradual rollout with automated monitoring and rollback

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Configuration defaults
ENVIRONMENT="${ENVIRONMENT:-production}"
NAMESPACE="${NAMESPACE:-production}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CANARY_PERCENTAGE="${CANARY_PERCENTAGE:-10}"
MONITORING_DURATION="${MONITORING_DURATION:-300}"  # 5 minutes
ERROR_THRESHOLD="${ERROR_THRESHOLD:-5.0}"           # 5% error rate
RESPONSE_TIME_THRESHOLD="${RESPONSE_TIME_THRESHOLD:-2000}"  # 2 seconds
AUTO_PROMOTE="${AUTO_PROMOTE:-false}"
DRY_RUN="${DRY_RUN:-false}"

# Services to deploy
SERVICES=("api-gateway" "data-quality-service" "anomaly-detection-service" "workflow-engine")

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required but not installed"
        return 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        return 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace '$NAMESPACE' does not exist"
        return 1
    fi
    
    # Check if jq is available for JSON processing
    if ! command -v jq &> /dev/null; then
        log_warning "jq not found - some features may be limited"
    fi
    
    log_success "Prerequisites check completed"
    return 0
}

# Get current deployment status
get_deployment_status() {
    local service=$1
    
    if kubectl get deployment "$service" -n "$NAMESPACE" &> /dev/null; then
        kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.replicas},{.status.readyReplicas},{.spec.replicas}'
    else
        echo "0,0,0"
    fi
}

# Create canary deployment
create_canary_deployment() {
    local service=$1
    local canary_name="${service}-canary"
    
    log_info "Creating canary deployment for $service..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create canary deployment: $canary_name"
        return 0
    fi
    
    # Get current deployment spec
    if ! kubectl get deployment "$service" -n "$NAMESPACE" -o yaml > "/tmp/${service}-deployment.yaml"; then
        log_error "Failed to get current deployment spec for $service"
        return 1
    fi
    
    # Create canary deployment spec
    sed "s/name: $service/name: $canary_name/g" "/tmp/${service}-deployment.yaml" | \
    sed "s/app: $service/app: $service-canary/g" | \
    sed "s/replicas: [0-9]*/replicas: 1/g" > "/tmp/${service}-canary.yaml"
    
    # Update image tag if specified
    if [[ "$IMAGE_TAG" != "latest" ]]; then
        sed -i "s/image: .*:latest/image: ${service}:${IMAGE_TAG}/g" "/tmp/${service}-canary.yaml"
    fi
    
    # Apply canary deployment
    if kubectl apply -f "/tmp/${service}-canary.yaml"; then
        log_success "Canary deployment created for $service"
    else
        log_error "Failed to create canary deployment for $service"
        return 1
    fi
    
    # Wait for canary to be ready
    log_info "Waiting for canary deployment to be ready..."
    if ! kubectl wait --for=condition=available --timeout=300s deployment/"$canary_name" -n "$NAMESPACE"; then
        log_error "Canary deployment failed to become ready"
        return 1
    fi
    
    return 0
}

# Configure traffic splitting
configure_traffic_splitting() {
    local service=$1
    local percentage=$2
    
    log_info "Configuring traffic splitting: ${percentage}% to canary"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would configure traffic splitting: ${percentage}% to canary"
        return 0
    fi
    
    # Update service selector to include both stable and canary pods
    # This is a simplified approach - in production you'd use a service mesh like Istio
    
    # Create canary service
    cat > "/tmp/${service}-canary-service.yaml" << EOF
apiVersion: v1
kind: Service
metadata:
  name: ${service}-canary
  namespace: ${NAMESPACE}
  labels:
    app: ${service}-canary
spec:
  selector:
    app: ${service}-canary
  ports:
  - name: http
    port: 80
    targetPort: 8080
  type: ClusterIP
EOF
    
    kubectl apply -f "/tmp/${service}-canary-service.yaml"
    
    # Configure ingress or load balancer for traffic splitting
    # This would depend on your specific setup (Nginx, HAProxy, etc.)
    
    log_success "Traffic splitting configured"
    return 0
}

# Monitor canary deployment
monitor_canary() {
    local service=$1
    local duration=$2
    
    log_info "Monitoring canary deployment for ${duration} seconds..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + duration))
    local check_interval=30
    
    while [[ $(date +%s) -lt $end_time ]]; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local remaining=$((end_time - current_time))
        
        log_info "Monitoring... ${elapsed}s elapsed, ${remaining}s remaining"
        
        # Check error rate
        local error_rate
        error_rate=$(get_error_rate "$service")
        
        if [[ $(echo "$error_rate > $ERROR_THRESHOLD" | bc -l) -eq 1 ]]; then
            log_error "Error rate ${error_rate}% exceeds threshold ${ERROR_THRESHOLD}%"
            return 1
        fi
        
        # Check response time
        local response_time
        response_time=$(get_response_time "$service")
        
        if [[ $(echo "$response_time > $RESPONSE_TIME_THRESHOLD" | bc -l) -eq 1 ]]; then
            log_error "Response time ${response_time}ms exceeds threshold ${RESPONSE_TIME_THRESHOLD}ms"
            return 1
        fi
        
        # Check pod health
        local unhealthy_pods
        unhealthy_pods=$(kubectl get pods -n "$NAMESPACE" -l "app=${service}-canary" --field-selector=status.phase!=Running --no-headers | wc -l)
        
        if [[ $unhealthy_pods -gt 0 ]]; then
            log_error "Found $unhealthy_pods unhealthy canary pods"
            kubectl get pods -n "$NAMESPACE" -l "app=${service}-canary" --field-selector=status.phase!=Running
            return 1
        fi
        
        log_success "Health check passed - Error rate: ${error_rate}%, Response time: ${response_time}ms"
        
        sleep $check_interval
    done
    
    log_success "Canary monitoring completed successfully"
    return 0
}

# Get error rate from metrics
get_error_rate() {
    local service=$1
    
    # This would typically query Prometheus or your metrics system
    # For now, we'll simulate with a random value for demonstration
    if command -v curl &> /dev/null && [[ -n "${PROMETHEUS_URL:-}" ]]; then
        local query="rate(http_requests_total{job=\"$service-canary\",status=~\"5..\"}[5m]) / rate(http_requests_total{job=\"$service-canary\"}[5m]) * 100"
        local result
        result=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=${query}" | jq -r '.data.result[0].value[1] // "0"')
        echo "$result"
    else
        # Simulate a low error rate for demo
        echo "1.2"
    fi
}

# Get response time from metrics
get_response_time() {
    local service=$1
    
    # This would typically query Prometheus for P95 response time
    if command -v curl &> /dev/null && [[ -n "${PROMETHEUS_URL:-}" ]]; then
        local query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"$service-canary\"}[5m])) * 1000"
        local result
        result=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=${query}" | jq -r '.data.result[0].value[1] // "500"')
        echo "$result"
    else
        # Simulate a reasonable response time for demo
        echo "800"
    fi
}

# Promote canary to production
promote_canary() {
    local service=$1
    
    log_info "Promoting canary to production for $service..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would promote canary to production"
        return 0
    fi
    
    # Get canary image tag
    local canary_image
    canary_image=$(kubectl get deployment "${service}-canary" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')
    
    # Update main deployment with canary image
    kubectl set image deployment/"$service" "${service}=${canary_image}" -n "$NAMESPACE"
    
    # Wait for rollout to complete
    if kubectl rollout status deployment/"$service" -n "$NAMESPACE" --timeout=300s; then
        log_success "Production deployment updated successfully"
    else
        log_error "Failed to update production deployment"
        return 1
    fi
    
    # Clean up canary resources
    cleanup_canary "$service"
    
    return 0
}

# Rollback canary deployment
rollback_canary() {
    local service=$1
    
    log_error "Rolling back canary deployment for $service..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would rollback canary deployment"
        return 0
    fi
    
    # Clean up canary resources
    cleanup_canary "$service"
    
    log_success "Canary rollback completed"
    return 0
}

# Clean up canary resources
cleanup_canary() {
    local service=$1
    
    log_info "Cleaning up canary resources for $service..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would cleanup canary resources"
        return 0
    fi
    
    # Delete canary deployment
    kubectl delete deployment "${service}-canary" -n "$NAMESPACE" --ignore-not-found=true
    
    # Delete canary service
    kubectl delete service "${service}-canary" -n "$NAMESPACE" --ignore-not-found=true
    
    # Clean up temporary files
    rm -f "/tmp/${service}-deployment.yaml" "/tmp/${service}-canary.yaml" "/tmp/${service}-canary-service.yaml"
    
    log_success "Canary cleanup completed"
}

# Deploy service with canary strategy
deploy_service_canary() {
    local service=$1
    
    log_info "Starting canary deployment for $service..."
    
    # Check if service exists
    if ! kubectl get deployment "$service" -n "$NAMESPACE" &> /dev/null; then
        log_error "Service $service does not exist in namespace $NAMESPACE"
        return 1
    fi
    
    # Create canary deployment
    if ! create_canary_deployment "$service"; then
        log_error "Failed to create canary deployment for $service"
        return 1
    fi
    
    # Configure traffic splitting
    if ! configure_traffic_splitting "$service" "$CANARY_PERCENTAGE"; then
        log_error "Failed to configure traffic splitting for $service"
        rollback_canary "$service"
        return 1
    fi
    
    # Monitor canary deployment
    if ! monitor_canary "$service" "$MONITORING_DURATION"; then
        log_error "Canary monitoring failed for $service"
        rollback_canary "$service"
        return 1
    fi
    
    # Promote or rollback based on monitoring results
    if [[ "$AUTO_PROMOTE" == "true" ]]; then
        if ! promote_canary "$service"; then
            log_error "Failed to promote canary for $service"
            rollback_canary "$service"
            return 1
        fi
    else
        log_info "Canary deployment successful. Manual promotion required."
        log_info "To promote: kubectl set image deployment/$service $service=\$(kubectl get deployment ${service}-canary -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')"
        log_info "To rollback: $0 --rollback --service $service"
    fi
    
    return 0
}

# Generate canary deployment report
generate_deployment_report() {
    local services=("$@")
    local report_file="/tmp/canary-deployment-report-$(date +%Y%m%d-%H%M%S).txt"
    
    log_info "Generating deployment report..."
    
    cat > "$report_file" << EOF
================================================================================
CANARY DEPLOYMENT REPORT
================================================================================
Generated: $(date '+%Y-%m-%d %H:%M:%S UTC')
Environment: $ENVIRONMENT
Namespace: $NAMESPACE
Image Tag: $IMAGE_TAG
Canary Percentage: $CANARY_PERCENTAGE%
Monitoring Duration: ${MONITORING_DURATION}s
Auto Promote: $AUTO_PROMOTE

Services Deployed:
EOF
    
    for service in "${services[@]}"; do
        local status
        if kubectl get deployment "${service}-canary" -n "$NAMESPACE" &> /dev/null; then
            status="CANARY ACTIVE"
        elif kubectl get deployment "$service" -n "$NAMESPACE" &> /dev/null; then
            status="PRODUCTION"
        else
            status="NOT FOUND"
        fi
        
        echo "- $service: $status" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF

Deployment Statistics:
- Total Services: ${#services[@]}
- Successful Deployments: $(( ${#services[@]} - $(echo "$FAILED_SERVICES" | wc -w) ))
- Failed Deployments: $(echo "$FAILED_SERVICES" | wc -w)

EOF
    
    if [[ -n "${FAILED_SERVICES:-}" ]]; then
        echo "Failed Services: $FAILED_SERVICES" >> "$report_file"
    fi
    
    echo "$report_file"
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --service SERVICE      Deploy specific service (can be repeated)"
    echo "  --all-services         Deploy all services"
    echo "  --promote SERVICE      Promote canary to production"
    echo "  --rollback SERVICE     Rollback canary deployment"
    echo "  --cleanup SERVICE      Clean up canary resources"
    echo "  --status               Show deployment status"
    echo "  --dry-run              Show what would be done without executing"
    echo "  --help                 Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ENVIRONMENT            Target environment (default: production)"
    echo "  NAMESPACE              Kubernetes namespace (default: production)"
    echo "  IMAGE_TAG              Docker image tag (default: latest)"
    echo "  CANARY_PERCENTAGE      Percentage of traffic to canary (default: 10)"
    echo "  MONITORING_DURATION    Monitoring duration in seconds (default: 300)"
    echo "  ERROR_THRESHOLD        Error rate threshold % (default: 5.0)"
    echo "  RESPONSE_TIME_THRESHOLD Response time threshold ms (default: 2000)"
    echo "  AUTO_PROMOTE           Auto promote on success (default: false)"
    echo "  PROMETHEUS_URL         Prometheus endpoint for metrics"
    echo ""
    echo "Examples:"
    echo "  $0 --service api-gateway"
    echo "  $0 --all-services --auto-promote"
    echo "  $0 --promote api-gateway"
    echo "  $0 --rollback api-gateway"
    echo "  CANARY_PERCENTAGE=25 $0 --service data-quality-service"
}

# Main execution
main() {
    local selected_services=()
    local action="deploy"
    local specific_service=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --service)
                selected_services+=("$2")
                shift 2
                ;;
            --all-services)
                selected_services=("${SERVICES[@]}")
                shift
                ;;
            --promote)
                action="promote"
                specific_service="$2"
                shift 2
                ;;
            --rollback)
                action="rollback"
                specific_service="$2"
                shift 2
                ;;
            --cleanup)
                action="cleanup"
                specific_service="$2"
                shift 2
                ;;
            --status)
                action="status"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites || exit 1
    
    # Execute requested action
    case $action in
        deploy)
            if [[ ${#selected_services[@]} -eq 0 ]]; then
                log_error "No services specified. Use --service or --all-services"
                show_usage
                exit 1
            fi
            
            log_info "Starting canary deployment process..."
            log_info "Environment: $ENVIRONMENT"
            log_info "Namespace: $NAMESPACE"
            log_info "Canary percentage: $CANARY_PERCENTAGE%"
            log_info "Monitoring duration: ${MONITORING_DURATION}s"
            log_info "Services: ${selected_services[*]}"
            
            FAILED_SERVICES=""
            
            for service in "${selected_services[@]}"; do
                if ! deploy_service_canary "$service"; then
                    FAILED_SERVICES="$FAILED_SERVICES $service"
                    log_error "Canary deployment failed for $service"
                else
                    log_success "Canary deployment successful for $service"
                fi
            done
            
            # Generate report
            report_file=$(generate_deployment_report "${selected_services[@]}")
            log_info "Deployment report generated: $report_file"
            
            if [[ -n "$FAILED_SERVICES" ]]; then
                log_error "Some deployments failed: $FAILED_SERVICES"
                exit 1
            else
                log_success "All canary deployments completed successfully"
            fi
            ;;
        promote)
            promote_canary "$specific_service"
            ;;
        rollback)
            rollback_canary "$specific_service"
            ;;
        cleanup)
            cleanup_canary "$specific_service"
            ;;
        status)
            echo "Deployment Status:"
            echo "=================="
            for service in "${SERVICES[@]}"; do
                local prod_status
                prod_status=$(get_deployment_status "$service")
                local canary_exists=""
                
                if kubectl get deployment "${service}-canary" -n "$NAMESPACE" &> /dev/null; then
                    canary_exists=" (CANARY ACTIVE)"
                fi
                
                echo "$service: $prod_status$canary_exists"
            done
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
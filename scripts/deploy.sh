#!/bin/bash

# =============================================================================
# Pynomaly API Production Deployment Script
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="$PROJECT_ROOT/deploy"

# Default values
ENVIRONMENT="production"
NAMESPACE="pynomaly-production"
IMAGE_TAG="latest"
SKIP_TESTS="false"
FORCE_DEPLOY="false"
DRY_RUN="false"
VERIFY_DEPLOYMENT="true"

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

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Pynomaly API to Kubernetes

OPTIONS:
    -e, --environment ENV       Deployment environment (default: production)
    -n, --namespace NAMESPACE   Kubernetes namespace (default: pynomaly-production)
    -t, --tag TAG              Docker image tag (default: latest)
    -s, --skip-tests           Skip pre-deployment tests
    -f, --force                Force deployment without confirmation
    -d, --dry-run              Show what would be deployed without executing
    -h, --help                 Show this help message

EXAMPLES:
    $0                                          # Deploy latest to production
    $0 -e staging -t v2.1.0                   # Deploy v2.1.0 to staging
    $0 -f -s                                   # Force deploy skipping tests
    $0 -d                                      # Dry run deployment

EOF
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
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -s|--skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        -f|--force)
            FORCE_DEPLOY="true"
            shift
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

validate_environment() {
    log "Validating deployment environment..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi
    
    # Check if kubectl context is set
    if ! kubectl config current-context &> /dev/null; then
        error "No kubectl context is set"
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        warning "Namespace '$NAMESPACE' does not exist. Creating..."
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl apply -f "$DEPLOY_DIR/kubernetes/namespace.yaml"
        fi
    fi
    
    # Validate environment-specific requirements
    case "$ENVIRONMENT" in
        production)
            # Additional production validations
            log "Validating production environment..."
            
            # Check for required secrets
            if ! kubectl get secret pynomaly-secrets -n "$NAMESPACE" &> /dev/null; then
                error "Production secrets not found in namespace '$NAMESPACE'"
            fi
            
            # Verify cluster resources
            local nodes_ready
            nodes_ready=$(kubectl get nodes --no-headers | grep -c "Ready")
            if [[ "$nodes_ready" -lt 3 ]]; then
                warning "Less than 3 nodes ready in cluster"
            fi
            ;;
        staging)
            log "Validating staging environment..."
            ;;
        development)
            log "Validating development environment..."
            ;;
        *)
            error "Unknown environment: $ENVIRONMENT"
            ;;
    esac
    
    success "Environment validation completed"
}

validate_image() {
    log "Validating Docker image..."
    
    local image_name="ghcr.io/pynomaly/pynomaly:$IMAGE_TAG"
    
    # Check if image exists in registry
    if command -v docker &> /dev/null; then
        if ! docker manifest inspect "$image_name" &> /dev/null; then
            error "Docker image '$image_name' not found in registry"
        fi
    else
        warning "Docker not available, skipping image validation"
    fi
    
    success "Image validation completed"
}

# =============================================================================
# PRE-DEPLOYMENT FUNCTIONS
# =============================================================================

run_pre_deployment_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warning "Skipping pre-deployment tests"
        return 0
    fi
    
    log "Running pre-deployment tests..."
    
    # API health check on current deployment
    if kubectl get deployment pynomaly-api -n "$NAMESPACE" &> /dev/null; then
        log "Checking current deployment health..."
        
        local service_ip
        service_ip=$(kubectl get service pynomaly-api-internal -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "")
        
        if [[ -n "$service_ip" ]]; then
            # Run health check via port-forward
            kubectl port-forward service/pynomaly-api-internal 8080:8000 -n "$NAMESPACE" &
            local port_forward_pid=$!
            
            sleep 5
            
            if curl -f http://localhost:8080/api/v1/health &> /dev/null; then
                success "Current deployment is healthy"
            else
                warning "Current deployment health check failed"
            fi
            
            kill $port_forward_pid 2>/dev/null || true
        fi
    fi
    
    # Run unit tests
    log "Running unit tests..."
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        cd "$PROJECT_ROOT"
        if command -v poetry &> /dev/null; then
            poetry run pytest tests/unit/ -x || error "Unit tests failed"
        else
            python -m pytest tests/unit/ -x || error "Unit tests failed"
        fi
    fi
    
    success "Pre-deployment tests completed"
}

backup_current_deployment() {
    log "Creating backup of current deployment..."
    
    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup current deployment manifests
    if kubectl get deployment pynomaly-api -n "$NAMESPACE" &> /dev/null; then
        kubectl get deployment pynomaly-api -n "$NAMESPACE" -o yaml > "$backup_dir/deployment.yaml"
        kubectl get service pynomaly-api-service -n "$NAMESPACE" -o yaml > "$backup_dir/service.yaml"
        kubectl get configmap pynomaly-config -n "$NAMESPACE" -o yaml > "$backup_dir/configmap.yaml"
        
        success "Backup created at $backup_dir"
    else
        log "No existing deployment to backup"
    fi
}

# =============================================================================
# DEPLOYMENT FUNCTIONS
# =============================================================================

deploy_database() {
    log "Deploying database components..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would deploy database"
        return 0
    fi
    
    # Apply database manifests
    kubectl apply -f "$DEPLOY_DIR/kubernetes/database-statefulset.yaml"
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=database -n "$NAMESPACE" --timeout=300s
    
    success "Database deployment completed"
}

deploy_redis() {
    log "Deploying Redis cache..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would deploy Redis"
        return 0
    fi
    
    # Apply Redis manifests
    kubectl apply -f "$DEPLOY_DIR/kubernetes/cache-deployment.yaml"
    
    # Wait for Redis to be ready
    log "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=cache -n "$NAMESPACE" --timeout=300s
    
    success "Redis deployment completed"
}

deploy_api() {
    log "Deploying API application..."
    
    # Update image tag in deployment manifest
    local temp_manifest="/tmp/pynomaly-deployment.yaml"
    local image_name="ghcr.io/pynomaly/pynomaly:$IMAGE_TAG"
    
    # Create temporary manifest with updated image
    sed "s|image: pynomaly:production-latest|image: $image_name|g" \
        "$DEPLOY_DIR/kubernetes/production-deployment.yaml" > "$temp_manifest"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would deploy API with image: $image_name"
        cat "$temp_manifest" | grep -A 2 -B 2 "image:"
        rm -f "$temp_manifest"
        return 0
    fi
    
    # Apply the deployment
    kubectl apply -f "$temp_manifest"
    
    # Wait for rollout to complete
    log "Waiting for API deployment rollout..."
    kubectl rollout status deployment/pynomaly-api -n "$NAMESPACE" --timeout=600s
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=api -n "$NAMESPACE" --timeout=300s
    
    # Clean up temporary manifest
    rm -f "$temp_manifest"
    
    success "API deployment completed"
}

deploy_monitoring() {
    log "Deploying monitoring components..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would deploy monitoring"
        return 0
    fi
    
    # Apply monitoring manifests if they exist
    if [[ -f "$DEPLOY_DIR/kubernetes/monitoring-deployment.yaml" ]]; then
        kubectl apply -f "$DEPLOY_DIR/kubernetes/monitoring-deployment.yaml"
        success "Monitoring deployment completed"
    else
        warning "Monitoring manifests not found, skipping"
    fi
}

# =============================================================================
# POST-DEPLOYMENT FUNCTIONS
# =============================================================================

verify_deployment() {
    if [[ "$VERIFY_DEPLOYMENT" == "false" ]] || [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    log "Verifying deployment..."
    
    # Check pod status
    local ready_pods
    ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api --no-headers | grep -c "Running" || echo "0")
    
    if [[ "$ready_pods" -lt 1 ]]; then
        error "No API pods are running"
    fi
    
    # Check service accessibility
    log "Checking service accessibility..."
    
    # Port forward for testing
    kubectl port-forward service/pynomaly-api-internal 8081:8000 -n "$NAMESPACE" &
    local port_forward_pid=$!
    
    sleep 10
    
    # Health check
    if curl -f http://localhost:8081/api/v1/health &> /dev/null; then
        success "Health check passed"
    else
        kill $port_forward_pid 2>/dev/null || true
        error "Health check failed"
    fi
    
    # API docs check
    if curl -f http://localhost:8081/api/v1/docs &> /dev/null; then
        success "API documentation accessible"
    else
        warning "API documentation check failed"
    fi
    
    # OpenAPI schema check
    local endpoint_count
    endpoint_count=$(curl -s http://localhost:8081/api/v1/openapi.json | jq '.paths | length' 2>/dev/null || echo "0")
    
    if [[ "$endpoint_count" -gt 100 ]]; then
        success "OpenAPI schema valid ($endpoint_count endpoints)"
    else
        warning "OpenAPI schema check failed or incomplete"
    fi
    
    # Clean up port forward
    kill $port_forward_pid 2>/dev/null || true
    
    success "Deployment verification completed"
}

post_deployment_tasks() {
    log "Running post-deployment tasks..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would run post-deployment tasks"
        return 0
    fi
    
    # Update deployment annotations
    kubectl annotate deployment pynomaly-api -n "$NAMESPACE" \
        deployment.kubernetes.io/revision-last-deployed="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        deployment.kubernetes.io/image-tag="$IMAGE_TAG" \
        --overwrite
    
    # Log deployment info
    log "Deployment Summary:"
    log "  Environment: $ENVIRONMENT"
    log "  Namespace: $NAMESPACE"
    log "  Image Tag: $IMAGE_TAG"
    log "  Replicas: $(kubectl get deployment pynomaly-api -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')"
    
    success "Post-deployment tasks completed"
}

# =============================================================================
# MAIN DEPLOYMENT WORKFLOW
# =============================================================================

main() {
    log "Starting Pynomaly API deployment..."
    log "Environment: $ENVIRONMENT"
    log "Namespace: $NAMESPACE"
    log "Image Tag: $IMAGE_TAG"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        warning "DRY RUN MODE - No changes will be made"
    fi
    
    # Confirmation for production
    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$FORCE_DEPLOY" == "false" ]] && [[ "$DRY_RUN" == "false" ]]; then
        echo -n "Are you sure you want to deploy to PRODUCTION? (yes/no): "
        read -r confirmation
        if [[ "$confirmation" != "yes" ]]; then
            log "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Pre-deployment phase
    validate_environment
    validate_image
    run_pre_deployment_tests
    backup_current_deployment
    
    # Deployment phase
    deploy_database
    deploy_redis
    deploy_api
    deploy_monitoring
    
    # Post-deployment phase
    verify_deployment
    post_deployment_tasks
    
    success "🚀 Deployment completed successfully!"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        log "Access your deployment:"
        log "  kubectl get services -n $NAMESPACE"
        log "  kubectl port-forward service/pynomaly-api-internal 8080:8000 -n $NAMESPACE"
        log "  curl http://localhost:8080/api/v1/health"
    fi
}

# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

cleanup() {
    log "Cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}

trap cleanup EXIT

# =============================================================================
# ENTRY POINT
# =============================================================================

main "$@"
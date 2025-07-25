#!/bin/bash

# Hexagonal Architecture Deployment Script
# Usage: ./deploy.sh [environment] [options]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_DIR="$DEPLOYMENT_DIR/compose"
KUBERNETES_DIR="$DEPLOYMENT_DIR/kubernetes"

# Default values
ENVIRONMENT="${1:-development}"
DEPLOYMENT_TYPE="${2:-docker}"
FORCE_REBUILD="${3:-false}"

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

# Help function
show_help() {
    cat << EOF
Hexagonal Architecture Deployment Script

Usage: $0 [environment] [deployment_type] [force_rebuild]

Arguments:
  environment      Target environment (development|staging|production) [default: development]
  deployment_type  Deployment type (docker|kubernetes) [default: docker]
  force_rebuild    Force rebuild of images (true|false) [default: false]

Examples:
  $0 development docker false
  $0 staging kubernetes true
  $0 production kubernetes false

Environment-specific features:
  development:
    - Hot reload enabled
    - Debug logging
    - Development databases
    - Local storage
  
  staging:
    - Production-like configuration
    - Reduced resource limits
    - Integration testing ready
  
  production:
    - Full resource allocation
    - SSL/TLS enabled
    - Horizontal auto-scaling
    - Monitoring and alerting

Options:
  -h, --help       Show this help message
  -v, --verbose    Enable verbose output
  --dry-run        Show what would be deployed without executing
  --health-check   Run health checks after deployment
EOF
}

# Validate environment
validate_environment() {
    case "$ENVIRONMENT" in
        development|staging|production)
            log_info "Environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

# Validate deployment type
validate_deployment_type() {
    case "$DEPLOYMENT_TYPE" in
        docker|kubernetes)
            log_info "Deployment type: $DEPLOYMENT_TYPE"
            ;;
        *)
            log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
            log_error "Valid types: docker, kubernetes"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker is not installed or not in PATH"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            log_error "Docker Compose is not installed or not in PATH"
            exit 1
        fi
        
        if ! docker info &> /dev/null; then
            log_error "Docker daemon is not running"
            exit 1
        fi
    fi
    
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is not installed or not in PATH"
            exit 1
        fi
        
        if ! command -v kustomize &> /dev/null; then
            log_error "kustomize is not installed or not in PATH"
            exit 1
        fi
        
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Cannot connect to Kubernetes cluster"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    if [[ "$FORCE_REBUILD" == "true" ]] || [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Building Docker images..."
        
        # Build all service images
        docker build -t hexagonal-architecture/data-quality:latest \
            -f "$DEPLOYMENT_DIR/docker/data-quality/Dockerfile" \
            "$DEPLOYMENT_DIR/../.."
        
        docker build -t hexagonal-architecture/mlops:latest \
            -f "$DEPLOYMENT_DIR/docker/mlops/Dockerfile" \
            "$DEPLOYMENT_DIR/../.."
        
        docker build -t hexagonal-architecture/machine-learning:latest \
            -f "$DEPLOYMENT_DIR/docker/machine-learning/Dockerfile" \
            "$DEPLOYMENT_DIR/../.."
        
        docker build -t hexagonal-architecture/anomaly-detection:latest \
            -f "$DEPLOYMENT_DIR/docker/anomaly-detection/Dockerfile" \
            "$DEPLOYMENT_DIR/../.."
        
        log_success "Docker images built successfully"
    else
        log_info "Skipping image build (use force_rebuild=true to rebuild)"
    fi
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    local compose_file="$COMPOSE_DIR/$ENVIRONMENT.yml"
    
    if [[ ! -f "$compose_file" ]]; then
        log_error "Compose file not found: $compose_file"
        exit 1
    fi
    
    # Create network if it doesn't exist
    docker network create hexagonal-architecture-network 2>/dev/null || true
    
    # Deploy services
    docker-compose -f "$compose_file" up -d --remove-orphans
    
    log_success "Docker deployment completed"
    
    # Show running services
    log_info "Running services:"
    docker-compose -f "$compose_file" ps
}

# Deploy with Kubernetes
deploy_kubernetes() {
    log_info "Deploying with Kubernetes..."
    
    local kustomize_dir="$KUBERNETES_DIR/overlays/$ENVIRONMENT"
    
    if [[ ! -d "$kustomize_dir" ]]; then
        log_error "Kustomize overlay not found: $kustomize_dir"
        exit 1
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -k "$kustomize_dir"
    
    log_success "Kubernetes deployment completed"
    
    # Show deployed resources
    log_info "Deployed resources:"
    kubectl get all -n "hexagonal-$ENVIRONMENT"
}

# Health check
run_health_check() {
    log_info "Running health checks..."
    
    local base_url
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        base_url="http://localhost"
    else
        base_url="http://api.hexagonal-architecture.com"
    fi
    
    # Check each service
    local services=(
        "data-quality:8000"
        "mlops-experiments:8001"
        "mlops-registry:8002"
        "ml-training:8004"
        "ml-prediction:8005"
        "anomaly-detection:8007"
    )
    
    for service in "${services[@]}"; do
        local name="${service%:*}"
        local port="${service#*:}"
        local url="$base_url:$port/health"
        
        if curl -f -s "$url" > /dev/null; then
            log_success "$name service is healthy"
        else
            log_warning "$name service health check failed"
        fi
    done
}

# Cleanup function
cleanup() {
    log_info "Cleaning up resources..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        local compose_file="$COMPOSE_DIR/$ENVIRONMENT.yml"
        if [[ -f "$compose_file" ]]; then
            docker-compose -f "$compose_file" down
        fi
    else
        local kustomize_dir="$KUBERNETES_DIR/overlays/$ENVIRONMENT"
        if [[ -d "$kustomize_dir" ]]; then
            kubectl delete -k "$kustomize_dir" || true
        fi
    fi
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            --dry-run)
                log_info "Dry run mode - showing what would be deployed"
                DRY_RUN=true
                shift
                ;;
            --health-check)
                HEALTH_CHECK=true
                shift
                ;;
            --cleanup)
                cleanup
                exit 0
                ;;
            *)
                shift
                ;;
        esac
    done
    
    log_info "Starting hexagonal architecture deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Deployment Type: $DEPLOYMENT_TYPE"
    
    # Validate inputs
    validate_environment
    validate_deployment_type
    
    # Check prerequisites
    check_prerequisites
    
    # Build images if needed
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        build_images
    fi
    
    # Deploy based on type
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        log_info "Dry run - would deploy $ENVIRONMENT environment using $DEPLOYMENT_TYPE"
        exit 0
    fi
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        deploy_docker
    else
        deploy_kubernetes
    fi
    
    # Run health checks if requested
    if [[ "${HEALTH_CHECK:-false}" == "true" ]]; then
        sleep 30  # Wait for services to start
        run_health_check
    fi
    
    log_success "Deployment completed successfully!"
    log_info "Access the services at:"
    
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        log_info "  Data Quality: http://localhost:8000"
        log_info "  MLOps Experiments: http://localhost:8001"
        log_info "  MLOps Registry: http://localhost:8002"
        log_info "  ML Training: http://localhost:8004"
        log_info "  ML Prediction: http://localhost:8005"
        log_info "  Anomaly Detection: http://localhost:8007"
        log_info "  Monitoring: http://localhost:3000 (Grafana)"
    else
        log_info "  All services: https://api.hexagonal-architecture.com"
        log_info "  Monitoring: https://api.hexagonal-architecture.com/dashboards"
    fi
}

# Handle script interruption
trap cleanup EXIT

# Run main function
main "$@"
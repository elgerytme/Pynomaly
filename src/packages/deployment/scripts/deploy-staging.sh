#!/bin/bash

# Staging Environment Deployment Script
# Usage: ./deploy-staging.sh [--build] [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
STAGING_DIR="$DEPLOYMENT_DIR/staging"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="hexagonal-staging"
BUILD_IMAGES=false
FORCE_DEPLOY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_IMAGES=true
            shift
            ;;
        --force)
            FORCE_DEPLOY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--build] [--force]"
            echo "  --build    Build Docker images before deployment"
            echo "  --force    Force deployment even if namespace exists"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker is available (if building)
    if [ "$BUILD_IMAGES" = true ] && ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

build_images() {
    if [ "$BUILD_IMAGES" = true ]; then
        log_info "Building Docker images for staging..."
        
        # Build all service images
        services=("data-quality" "machine-learning" "mlops" "anomaly-detection")
        
        for service in "${services[@]}"; do
            log_info "Building $service image..."
            docker build -t "hexagonal-architecture/$service:staging" \
                -f "$DEPLOYMENT_DIR/docker/$service/Dockerfile" \
                --target production \
                "$DEPLOYMENT_DIR/../$service" || {
                log_error "Failed to build $service image"
                exit 1
            }
        done
        
        log_success "All Docker images built successfully"
    else
        log_info "Skipping image build (use --build to build images)"
    fi
}

check_namespace() {
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        if [ "$FORCE_DEPLOY" = false ]; then
            log_warning "Namespace $NAMESPACE already exists. Use --force to redeploy."
            read -p "Do you want to continue and update the existing deployment? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Deployment cancelled"
                exit 0
            fi
        else
            log_info "Force deployment: updating existing namespace"
        fi
    fi
}

deploy_to_staging() {
    log_info "Deploying to staging environment..."
    
    # Apply the staging deployment
    kubectl apply -f "$STAGING_DIR/staging-deployment.yaml" || {
        log_error "Failed to apply staging deployment"
        exit 1
    }
    
    log_success "Staging deployment applied"
}

wait_for_rollout() {
    log_info "Waiting for deployments to be ready..."
    
    deployments=("data-quality-staging" "machine-learning-staging" "mlops-staging" "anomaly-detection-staging")
    
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for $deployment to be ready..."
        kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout=300s || {
            log_error "Deployment $deployment failed to become ready"
            kubectl get pods -n "$NAMESPACE" -l app="${deployment%-staging}"
            exit 1
        }
    done
    
    log_success "All deployments are ready"
}

verify_deployment() {
    log_info "Verifying staging deployment..."
    
    # Check pod status
    log_info "Pod status:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # Check service status
    log_info "Service status:"
    kubectl get services -n "$NAMESPACE"
    
    # Check ingress status
    log_info "Ingress status:"
    kubectl get ingress -n "$NAMESPACE"
    
    # Perform basic health checks
    log_info "Performing health checks..."
    
    services=("data-quality-service" "machine-learning-service" "mlops-service" "anomaly-detection-service")
    
    for service in "${services[@]}"; do
        # Port forward and check health endpoint
        kubectl port-forward -n "$NAMESPACE" "service/$service" 8080:80 &
        PF_PID=$!
        sleep 2
        
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "$service health check passed"
        else
            log_warning "$service health check failed (this may be expected if health endpoint is not implemented)"
        fi
        
        kill $PF_PID 2>/dev/null || true
        sleep 1
    done
}

display_access_info() {
    log_info "Deployment Summary:"
    echo
    echo "Namespace: $NAMESPACE"
    echo "Services deployed:"
    echo "  • Data Quality Service"
    echo "  • Machine Learning Service" 
    echo "  • MLOps Service"
    echo "  • Anomaly Detection Service"
    echo
    echo "Access URLs (after configuring ingress):"
    echo "  • Data Quality: http://staging.hexagonal-arch.local/data-quality"
    echo "  • Machine Learning: http://staging.hexagonal-arch.local/machine-learning"
    echo "  • MLOps: http://staging.hexagonal-arch.local/mlops"
    echo "  • Anomaly Detection: http://staging.hexagonal-arch.local/anomaly-detection"
    echo
    echo "To access services locally:"
    echo "  kubectl port-forward -n $NAMESPACE service/data-quality-service 8000:80"
    echo "  kubectl port-forward -n $NAMESPACE service/machine-learning-service 8001:80"
    echo "  kubectl port-forward -n $NAMESPACE service/mlops-service 8002:80"
    echo "  kubectl port-forward -n $NAMESPACE service/anomaly-detection-service 8003:80"
    echo
    echo "To monitor deployment:"
    echo "  kubectl get pods -n $NAMESPACE -w"
    echo "  kubectl logs -n $NAMESPACE -l environment=staging -f"
}

cleanup_on_failure() {
    if [ $? -ne 0 ]; then
        log_error "Deployment failed. Cleaning up..."
        kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    fi
}

main() {
    trap cleanup_on_failure ERR
    
    log_info "Starting staging deployment..."
    echo "Configuration:"
    echo "  Build images: $BUILD_IMAGES"
    echo "  Force deploy: $FORCE_DEPLOY"
    echo "  Namespace: $NAMESPACE"
    echo
    
    check_prerequisites
    build_images
    check_namespace
    deploy_to_staging
    wait_for_rollout
    verify_deployment
    display_access_info
    
    log_success "Staging deployment completed successfully!"
}

main "$@"
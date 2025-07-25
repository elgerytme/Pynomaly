#!/bin/bash

# MLOps Marketplace Deployment Script
# This script handles deployment of the marketplace to various environments

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-marketplace.company.com}"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

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

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if kubectl is installed (for Kubernetes deployments)
    if [[ "$ENVIRONMENT" == "kubernetes" ]] && ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is installed (for Helm deployments)
    if [[ "$1" == "helm" ]] && ! command -v helm &> /dev/null; then
        log_error "Helm is not installed or not in PATH"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Function to build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_DIR"
    
    # Build main API image
    log_info "Building marketplace API image..."
    docker build -t "${DOCKER_REGISTRY}/marketplace-api:${VERSION}" \
        --target production \
        --build-arg VERSION="$VERSION" \
        .
    
    # Build worker image
    log_info "Building marketplace worker image..."
    docker build -t "${DOCKER_REGISTRY}/marketplace-worker:${VERSION}" \
        -f Dockerfile.worker \
        --build-arg VERSION="$VERSION" \
        .
    
    # Build web interface image (if exists)
    if [[ -f "Dockerfile.web" ]]; then
        log_info "Building marketplace web image..."
        docker build -t "${DOCKER_REGISTRY}/marketplace-web:${VERSION}" \
            -f Dockerfile.web \
            --build-arg VERSION="$VERSION" \
            .
    fi
    
    log_success "Docker images built successfully"
}

# Function to push images to registry
push_images() {
    log_info "Pushing images to registry..."
    
    # Push API image
    docker push "${DOCKER_REGISTRY}/marketplace-api:${VERSION}"
    
    # Push worker image
    docker push "${DOCKER_REGISTRY}/marketplace-worker:${VERSION}"
    
    # Push web image if it exists
    if docker images "${DOCKER_REGISTRY}/marketplace-web:${VERSION}" --format "table {{.Repository}}" | grep -q marketplace-web; then
        docker push "${DOCKER_REGISTRY}/marketplace-web:${VERSION}"
    fi
    
    log_success "Images pushed to registry"
}

# Function to deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_DIR"
    
    # Create environment-specific compose file
    COMPOSE_FILE="docker-compose.yml"
    if [[ -f "docker-compose.${ENVIRONMENT}.yml" ]]; then
        COMPOSE_FILE="docker-compose.yml:docker-compose.${ENVIRONMENT}.yml"
    fi
    
    # Stop existing containers
    docker-compose -f "$COMPOSE_FILE" down
    
    # Pull latest images
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check health
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "healthy\|Up"; then
        log_success "Services are running and healthy"
    else
        log_error "Some services failed to start properly"
        docker-compose -f "$COMPOSE_FILE" logs
        exit 1
    fi
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    cd "$PROJECT_DIR"
    
    # Apply namespace
    kubectl apply -f deploy/k8s/namespace.yaml
    
    # Apply configmaps and secrets
    kubectl apply -f deploy/k8s/config/
    
    # Apply database migrations job
    kubectl apply -f deploy/k8s/jobs/db-migration.yaml
    
    # Wait for migration to complete
    kubectl wait --for=condition=complete job/db-migration --timeout=300s
    
    # Apply deployments
    kubectl apply -f deploy/k8s/deployments/
    
    # Apply services
    kubectl apply -f deploy/k8s/services/
    
    # Apply ingress
    kubectl apply -f deploy/k8s/ingress/
    
    # Wait for rollout to complete
    kubectl rollout status deployment/marketplace-api
    kubectl rollout status deployment/marketplace-worker
    
    log_success "Kubernetes deployment completed"
}

# Function to deploy with Helm
deploy_helm() {
    log_info "Deploying with Helm..."
    
    cd "$PROJECT_DIR"
    
    HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-marketplace}"
    HELM_NAMESPACE="${HELM_NAMESPACE:-marketplace}"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$HELM_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Update Helm dependencies
    helm dependency update helm/marketplace/
    
    # Deploy or upgrade
    helm upgrade --install "$HELM_RELEASE_NAME" helm/marketplace/ \
        --namespace "$HELM_NAMESPACE" \
        --set image.tag="$VERSION" \
        --set environment="$ENVIRONMENT" \
        --timeout 10m \
        --wait
    
    log_success "Helm deployment completed"
}

# Function to run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    case "$ENVIRONMENT" in
        "docker-compose")
            docker-compose exec marketplace-api mlops-marketplace db migrate
            ;;
        "kubernetes")
            kubectl exec -it deployment/marketplace-api -- mlops-marketplace db migrate
            ;;
        "helm")
            kubectl exec -it -n "$HELM_NAMESPACE" deployment/"$HELM_RELEASE_NAME"-api -- mlops-marketplace db migrate
            ;;
        *)
            log_warning "Manual migration required for environment: $ENVIRONMENT"
            ;;
    esac
    
    log_success "Database migrations completed"
}

# Function to run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Wait a bit for services to stabilize
    sleep 10
    
    case "$ENVIRONMENT" in
        "docker-compose")
            HEALTH_URL="http://localhost:8000/health"
            ;;
        "kubernetes"|"helm")
            # Port forward to access service
            kubectl port-forward service/marketplace-api 8000:80 &
            PF_PID=$!
            sleep 5
            HEALTH_URL="http://localhost:8000/health"
            ;;
        *)
            log_warning "Smoke tests not configured for environment: $ENVIRONMENT"
            return 0
            ;;
    esac
    
    # Test health endpoint
    if curl -f "$HEALTH_URL" > /dev/null 2>&1; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        [[ -n "${PF_PID:-}" ]] && kill $PF_PID
        exit 1
    fi
    
    # Test API endpoints
    if curl -f "$HEALTH_URL/../api/v1/solutions/search?limit=1" > /dev/null 2>&1; then
        log_success "API smoke test passed"
    else
        log_error "API smoke test failed"
        [[ -n "${PF_PID:-}" ]] && kill $PF_PID
        exit 1
    fi
    
    # Clean up port forward
    [[ -n "${PF_PID:-}" ]] && kill $PF_PID
    
    log_success "Smoke tests passed"
}

# Function to display deployment status
show_status() {
    log_info "Deployment Status:"
    
    case "$ENVIRONMENT" in
        "docker-compose")
            docker-compose ps
            ;;
        "kubernetes")
            kubectl get pods,services,ingress
            ;;
        "helm")
            helm status "$HELM_RELEASE_NAME" -n "$HELM_NAMESPACE"
            kubectl get pods,services,ingress -n "$HELM_NAMESPACE"
            ;;
    esac
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    build           Build Docker images
    deploy          Deploy the marketplace
    migrate         Run database migrations
    test            Run smoke tests
    status          Show deployment status
    rollback        Rollback to previous version

Options:
    -e, --environment ENV    Deployment environment (docker-compose|kubernetes|helm)
    -v, --version VERSION    Image version tag (default: latest)
    -r, --registry REGISTRY  Docker registry URL
    -h, --help              Show this help message

Environment Variables:
    DOCKER_REGISTRY         Docker registry URL
    VERSION                 Image version tag
    ENVIRONMENT            Deployment environment
    HELM_RELEASE_NAME      Helm release name (for helm deployments)
    HELM_NAMESPACE         Helm namespace (for helm deployments)

Examples:
    $0 -e docker-compose deploy
    $0 -e kubernetes -v v1.2.3 deploy
    $0 -e helm deploy
    $0 build
    $0 test
EOF
}

# Function to rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    case "$ENVIRONMENT" in
        "kubernetes")
            kubectl rollout undo deployment/marketplace-api
            kubectl rollout undo deployment/marketplace-worker
            kubectl rollout status deployment/marketplace-api
            kubectl rollout status deployment/marketplace-worker
            ;;
        "helm")
            helm rollback "$HELM_RELEASE_NAME" -n "$HELM_NAMESPACE"
            ;;
        *)
            log_error "Rollback not supported for environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    log_success "Rollback completed"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        build|deploy|migrate|test|status|rollback)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate command
if [[ -z "${COMMAND:-}" ]]; then
    log_error "No command specified"
    usage
    exit 1
fi

# Main execution
main() {
    log_info "Starting MLOps Marketplace deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    log_info "Registry: $DOCKER_REGISTRY"
    
    case "$COMMAND" in
        build)
            check_prerequisites
            build_images
            ;;
        deploy)
            check_prerequisites "$ENVIRONMENT"
            build_images
            push_images
            case "$ENVIRONMENT" in
                "docker-compose")
                    deploy_docker_compose
                    ;;
                "kubernetes")
                    deploy_kubernetes
                    ;;
                "helm")
                    deploy_helm
                    ;;
                *)
                    log_error "Unsupported environment: $ENVIRONMENT"
                    exit 1
                    ;;
            esac
            run_migrations
            run_smoke_tests
            show_status
            ;;
        migrate)
            run_migrations
            ;;
        test)
            run_smoke_tests
            ;;
        status)
            show_status
            ;;
        rollback)
            rollback_deployment
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            usage
            exit 1
            ;;
    esac
    
    log_success "Deployment script completed successfully!"
}

# Run main function
main